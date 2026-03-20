from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms, models
from PIL import Image
from transformers import ViTModel, ViTConfig
import torch.nn as nn
import torch.nn.functional as F
import gdown

app = Flask(__name__)
app.secret_key = "retinal_secret_key"

UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "predictions_txt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 224

main_classes = ["Normal", "Glaucoma", "DR"]
stage_classes = ["Mild", "Moderate", "Severe", "Proliferative"]

# ================= MODEL =================

class UnifiedHybridModel(nn.Module):
    def __init__(self, num_main_classes=3, num_stage_classes=4, fusion_dim=512):
        super().__init__()

        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()
        resnet_dim = 2048

        vit_config = ViTConfig(
            image_size=image_size,
            num_channels=3,
            hidden_size=768,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=1024
        )

        self.vit = ViTModel(vit_config)
        vit_dim = vit_config.hidden_size

        self.fusion = nn.Sequential(
            nn.Linear(resnet_dim + vit_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.main_head = nn.Sequential(
            nn.Linear(fusion_dim,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,num_main_classes)
        )

        self.stage_head = nn.Sequential(
            nn.Linear(fusion_dim,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,num_stage_classes)
        )

    def forward(self,x):
        res_feat = self.resnet(x)
        vit_feat = self.vit(x).pooler_output
        fused = torch.cat((res_feat,vit_feat),dim=1)
        fused = self.fusion(fused)
        main_out = self.main_head(fused)
        stage_out = self.stage_head(fused)
        return main_out, stage_out

# ================= DOWNLOAD MODEL =================

MODEL_PATH = "unified_hybrid_model.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://drive.google.com/uc?id=1-KxXQiO6rvraZFiapa_opvuDy7la25l8"
    gdown.download(url, MODEL_PATH, quiet=False)

# ================= LOAD MODEL =================

model = UnifiedHybridModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ================= IMAGE TRANSFORM =================

transform = transforms.Compose([
    transforms.Resize((image_size,image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ================= LOGIN =================

@app.route("/", methods=["GET","POST"])
def login():

    if request.method=="POST":

        username = request.form["username"]
        password = request.form["password"]

        if username=="admin" and password=="1234":
            session["user"]=username
            return redirect(url_for("info"))
        else:
            return render_template("login.html", error="Invalid login")

    return render_template("login.html")

# ================= INFO PAGE =================

@app.route("/info")
def info():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("info.html")

# ================= PREDICTION =================

@app.route("/predict", methods=["GET","POST"])
def predict():

    if "user" not in session:
        return redirect(url_for("login"))

    result=None
    img_filename=None

    if request.method=="POST":

        file=request.files["file"]
        filename=secure_filename(file.filename)

        img_path=os.path.join(app.config["UPLOAD_FOLDER"],filename)
        file.save(img_path)

        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():

            main_out, stage_out = model(image_tensor)

            main_prob = F.softmax(main_out, dim=1).cpu().numpy()[0]
            main_pred = main_classes[main_prob.argmax()]
            disease_prob = float(main_prob.max())

            if main_pred=="DR":
                stage_pred = stage_classes[torch.argmax(stage_out,dim=1).item()]
                message=f"Detected Diabetic Retinopathy ({stage_pred})"
            elif main_pred=="Glaucoma":
                stage_pred="Not Applicable"
                message="Detected Glaucoma — Not DR"
            else:
                stage_pred="Not Applicable"
                message="Normal Eye"

        now=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        txt_name=f"{filename}_{now}.txt"

        with open(os.path.join(RESULTS_FOLDER,txt_name),"w") as f:
            f.write(f"Disease: {main_pred}\n")
            f.write(f"Probability: {disease_prob}\n")
            f.write(f"DR Stage: {stage_pred}\n")

        result={
            "main_pred":main_pred,
            "main_prob":disease_prob,
            "stage_pred":stage_pred,
            "message":message
        }

        img_filename=filename

    return render_template("predict.html", result=result, img_filename=img_filename)

# ================= DISPLAY IMAGE =================

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ================= LOGOUT =================

@app.route("/logout")
def logout():
    session.pop("user",None)
    return redirect(url_for("login"))

# ================= RUN =================

if __name__=="__main__":
    app.run()