"""Launch the newest verified workflow while preserving historical tabs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'DataSimulation'),str(ROOT/'WeldDefectDetection')]
def research_app():
    from raised_crown.campaign import OUT as crown,read
    from raised_acquisition.campaign import OUT as acquisition
    def ready(folder):
        return ((folder/'audit.json').exists() and (folder/'gui_checks.json').exists()
            and read(folder/'audit.json').get('status')=='passed')
    if ready(acquisition):
        from raised_acquisition.gui import create_app
    elif ready(crown):
        from raised_crown.gui import create_app
    else:
        from inspection.workbench import create_app
    from inspection.final_gui import attach
    return attach(create_app(ROOT/'DataSimulation/config'))

if __name__=='__main__':
    if '--research' in sys.argv:
        research_app().run()
    else:
        from inspection.studio_gui import create_app
        app=create_app()
        if '--open' in sys.argv or '--generation' in sys.argv:
            import argparse
            parser=argparse.ArgumentParser(description='Weld Vision')
            choice=parser.add_mutually_exclusive_group(required=True)
            choice.add_argument('--open',type=Path)
            choice.add_argument('--generation',type=Path)
            selected=parser.parse_args()
            app.root.after(200,lambda:app.select_file(selected.open) if selected.open else app.load_generated(selected.generation))
        app.run()
