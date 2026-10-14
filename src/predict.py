from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    conf=0.25,
    save=True,
    project="runs",
    name="smoke_predict"
)

print("Smoke test completed.")
print(f"Number of result objects: {len(results)}")
