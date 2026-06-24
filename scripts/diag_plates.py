import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from api.core.frame_source import FileFrameSource
from api.core.models import load_models
from api.core.pipeline_core import process_frames

def main():
    video_path = str(ROOT / "data/realworld-videos/chunks/hcm_night_01.mp4")
    models = load_models()
    source = FileFrameSource(video_path)

    plates = []

    def emit(event):
        if event.get("type") == "vehicle":
            plates.append(event)
            print(f"EMITTED: TID={event.get('recognition_id')} PLATE={event.get('plate_text')} CONF={event.get('plate_confidence')}")
        elif event.get("type") == "rejected_vehicle":
            print(f"REJECTED: TID={event.get('recognition_id')} PLATE={event.get('plate_text')} REASON={event.get('reason')} ROUTES={event.get('routes')}")

    process_frames(source, emit, models, ocr_backend="default")

if __name__ == "__main__":
    main()
