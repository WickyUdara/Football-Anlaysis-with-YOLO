from ultralytics import YOLO

model = YOLO("models/best.pt")  

results = model.predict(
    source='input_videos/video1.mp4',
    save=True,             
    device='cuda',         
    show=False,            
    conf=0.25              
)

print(results[0])
print('==============================')

for box in results[0].boxes:
    print(box)
