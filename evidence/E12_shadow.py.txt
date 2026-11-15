"""Nonlocal visibility for the existing parallel-direction detector convention.

Horizons are computed on a physical polar grid from the native Cartesian height
field, then interpolated to detector quadrature points. Resolution is checked
against direct rays in the pilot. This is geometric visibility, not transport.
"""
from unittest.mock import patch
import numpy as np
from scipy.ndimage import map_coordinates
import cv2
from bse_renderer.renderer import quadrant_detector as q

class VisibilityQuadrature:
    def __init__(self,height,detector,radial_step=.1,angular_stride=4,ray_step=.1,base=None):
        self.height=height;self.detector=detector;self.radial_step=radial_step
        self.angular_stride=angular_stride;self.ray_step=ray_step
        self.origin=(height.shape[0]-1)*.025
        self.us=np.arange(-6,6+radial_step/2,radial_step)
        self.theta=np.arange(0,5024,angular_stride)*2*np.pi/5024
        self.x=(40+self.us[:,None])*np.cos(self.theta)
        self.y=(40+self.us[:,None])*np.sin(self.theta)
        self.z=self.sample(self.x,self.y);self.maps={}
        # Maximum horizon beyond this distance cannot exceed the shallowest
        # detector elevation, because the global surface maximum is bounded.
        max_tan=detector.outer_radius_mm/detector.height_mm
        self.reach=float((height.max()-height.min()+.02)*max_tan+ray_step)
        self.current=None
        self.base=base;self.changed_regions=[];self.output=None
        if base is not None:
            assert (base.radial_step,base.angular_stride,base.ray_step)==(radial_step,angular_stride,ray_step)
            assert base.detector==detector
            changed=(height!=base.height).astype('u1')
            _,_,stats,_=cv2.connectedComponentsWithStats(changed,8)
            for x,y,w,h,area in stats[1:]:
                self.changed_regions.append(((x+(w-1)/2)*.05-self.origin,(y+(h-1)/2)*.05-self.origin,np.hypot(w,h)*.025+.3))

    def sample(self,x,y):
        return map_coordinates(self.height,[(y+self.origin)/.05,(x+self.origin)/.05],order=1,mode='constant',cval=0.)

    def horizon(self,az):
        key=round(float(az%(2*np.pi)),12)
        if key not in self.maps:
            if self.base is None:
                horizon=np.full(self.z.shape,-np.inf,dtype='f4');active=np.ones(self.z.shape,bool)
            else:
                horizon=self.base.horizon(az).copy();active=np.zeros(self.z.shape,bool)
                for cx,cy,radius in self.changed_regions:
                    dx=cx-self.x;dy=cy-self.y
                    along=dx*np.cos(az)+dy*np.sin(az);across=abs(-dx*np.sin(az)+dy*np.cos(az))
                    active|=(along>=-radius)&(along<=max(self.reach,self.base.reach)+radius)&(across<=radius)
            x=self.x[active];y=self.y[active];z=self.z[active]
            updated=np.full(x.shape,-np.inf,dtype='f4')
            for step in np.arange(self.ray_step,self.reach+self.ray_step/2,self.ray_step):
                if not len(x):break
                shifted=self.sample(x+step*np.cos(az),y+step*np.sin(az))
                updated=np.maximum(updated,(shifted-z)/step)
            horizon[active]=updated
            self.maps[key]=horizon
        return self.maps[key]

    def at(self,az):
        # Explicit periodic angular padding; radial queries stay within +/-6 mm.
        m=self.horizon(az);wrapped=np.pad(m,((0,0),(1,1)),mode='wrap')
        u,theta=self.current
        return map_coordinates(wrapped,[(u+6)/self.radial_step,(theta%(2*np.pi))*5024/(2*np.pi*self.angular_stride)+1],order=1,mode='nearest')

    def integrated(self,normals,alpha,tilt_azimuth,center_rad,phi_inner,phi_outer,b_coeff,
                   material_n,directional_gain,order_phi,order_varphi,angular_density=None):
        pn,pw=q._gauss_legendre_interval(phi_inner,phi_outer,order_phi)
        az,aw=q._gauss_legendre_interval(center_rad-np.pi/4,center_rad+np.pi/4,order_varphi)
        phi=pn[:,None,None,None];azi=az[None,:,None,None]
        dx=np.cos(azi)*np.sin(phi);dy=np.sin(azi)*np.sin(phi);dz=np.cos(phi)
        mu=normals[None,None,...,0]*dx+normals[None,None,...,1]*dy+normals[None,None,...,2]*dz
        tangent=normals*normals[...,2,None];tangent[...,2]-=1
        length=np.linalg.norm(tangent,axis=-1,keepdims=True)
        tangent=np.divide(tangent,length,out=np.zeros_like(tangent),where=length>1e-10)
        tangent=np.where(length>1e-10,tangent,np.array([1.,0.,0.]));tangent2=np.cross(normals,tangent)
        def project(axis):return axis[None,None,...,0]*dx+axis[None,None,...,1]*dy+axis[None,None,...,2]*dz
        local_azi=np.arctan2(project(tangent2),project(tangent));view=np.clip(mu,0,1)
        b=np.asarray(b_coeff)
        if b.ndim:b=b[None,None,...]
        emission=q.emission_distribution(np.arccos(view),local_azi,alpha[None,None,...],b)
        if angular_density is not None:raise ValueError('New visibility study uses explicit Lambertian fallback only')
        gain=float(np.clip(directional_gain,.2,2.5))
        directional=(1-gain)+gain*(.58*emission+.42*view**max(material_n,1.))
        horizons=np.stack([self.at(a) for a in az])[None,:,None,:]
        visible=(1/np.tan(phi)>=horizons)
        weights=pw[:,None]*aw[None,:]*np.sin(pn)[:,None]
        return np.sum(np.clip(directional,.05,None)*(mu>0)*visible*weights[...,None,None],axis=(0,1))

    def compute(self,cache=None,full=False,**kw):
        h,w=kw['normals'].shape[:2];axis=(np.arange(h)-(h-1)/2)*.05
        yy,xx=np.meshgrid(axis,axis,indexing='ij');u=np.hypot(xx,yy)-40;theta=np.arctan2(yy,xx)
        roi=abs(u)<=6
        if self.base is not None and self.base.output is not None:
            affected=np.zeros((h,w),bool)
            for cx,cy,radius in self.changed_regions:
                affected|=(xx-cx)**2+(yy-cy)**2<=(max(self.reach,self.base.reach)+radius+.5)**2
            ids=np.flatnonzero(roi&affected);out=self.base.output.copy()
        else:ids=np.flatnonzero(roi);out=np.zeros((h*w,4),'f4')
        with patch.object(q,'_integrated_quadrant_directional_term',self.integrated):
            for start in range(0,len(ids),8192):
                ix=ids[start:start+8192];self.current=(u.ravel()[ix],theta.ravel()[ix])
                part={k:(v.reshape((-1,)+v.shape[2:])[ix][None] if isinstance(v,np.ndarray) and v.shape[:2]==(h,w) else v) for k,v in kw.items()}
                out[ix]=q.compute_quadrant_signals(**part)[0]
        self.output=out.copy()
        return out.reshape(h,w,4)
