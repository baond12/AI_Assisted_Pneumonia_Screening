import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# --- 1. CẤU HÌNH & LOAD MODEL ---
def get_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    try:
        # Đường dẫn file model của bạn
        model_path = r'C:\Users\BaoND12\Documents\AI_camera\2026-02-05\pneumonia_resnet18_final.pth'
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    except FileNotFoundError:
        return None
    return model

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

# --- 2. GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="AI Doctor - Pneumonia Detection", page_icon="🩻", layout="wide")

st.title("Hệ Thống Chẩn Đoán Viêm Phổi qua X-Quang")
st.markdown("""
**Hệ thống hỗ trợ bác sĩ phát hiện dấu hiệu Viêm phổi (Pneumonia) từ ảnh X-Quang ngực.**
* **Mô hình:** CNN (ResNet18) - Transfer Learning.
* **Cơ chế:** Safety-First (Ưu tiên độ nhạy cao).
""")

# --- SIDEBAR: CẤU HÌNH NGƯỠNG ---
st.sidebar.header("Cấu hình Hệ thống")
uploaded_file = st.sidebar.file_uploader("Chọn file ảnh (.jpg, .png, .jpeg)", type=["jpg", "png", "jpeg"])

st.sidebar.markdown("---")
st.sidebar.subheader("🎚️ Ngưỡng Quyết định (Threshold)")
# Thanh trượt để điều chỉnh ngưỡng. Mặc định để 0.1 như bạn muốn.
threshold = st.sidebar.slider(
    "Độ nhạy sàng lọc (Threshold)", 
    min_value=0.0, max_value=1.0, value=0.1, step=0.05,
    help="Ngưỡng xác suất để máy báo Viêm phổi. Ngưỡng càng thấp, máy càng nhạy (ít bỏ sót bệnh)."
)
st.sidebar.info(f"Hiện tại: Nếu xác suất Viêm phổi >= **{threshold*100:.0f}%** --> Báo **DƯƠNG TÍNH**.")

# Load model
model = get_model()

if model is None:
    st.error(f"⚠️ Không tìm thấy file model tại: `C:\\Users\\BaoND12\\Documents\\AI_camera\\2026-02-05\\pneumonia_resnet18_final.pth`")
else:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh X-Quang bệnh nhân")
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Ảnh đã tải lên', use_container_width=True)
            
            # --- 3. DỰ ĐOÁN ---
            img_tensor = preprocess(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
            # Lấy xác suất cụ thể
            pneumonia_prob_val = probs[0][1].item() # Xác suất lớp 1 (Pneumonia)
            normal_prob_val = probs[0][0].item()    # Xác suất lớp 0 (Normal)
            
            # --- LOGIC QUAN TRỌNG: ÁP DỤNG NGƯỠNG ---
            if pneumonia_prob_val >= threshold:
                pred_label = 'PNEUMONIA'
            else:
                pred_label = 'NORMAL'
            
            # Chuyển sang % để hiển thị
            pneumonia_pct = pneumonia_prob_val * 100
            normal_pct = normal_prob_val * 100
            
    with col2:
        st.subheader("Kết quả chẩn đoán từ AI")
        if uploaded_file is not None:
            st.markdown("---")
            
            # Logic hiển thị
            if pred_label == 'PNEUMONIA':
                st.error(f"🚨 **DƯƠNG TÍNH: PHÁT HIỆN VIÊM PHỔI**")
                # Hiển thị xác suất Viêm phổi (để so sánh với ngưỡng)
                st.metric(label="Xác suất Viêm phổi", value=f"{pneumonia_pct:.2f}%", delta=f"Vượt ngưỡng {threshold*100}%")
                st.progress(int(pneumonia_pct))
                
                if threshold < 0.5:
                    st.warning(f"⚠️ **Lưu ý:** Hệ thống đang chạy ở chế độ nhạy cao (Ngưỡng {threshold}). Kết quả này cần bác sĩ kiểm tra lại để loại trừ khả năng báo nhầm.")
                else:
                    st.warning("⚠️ **Khuyến nghị:** Cần bác sĩ kiểm tra phổi ngay lập tức.")
            
            else:
                st.success(f"✅ **BÌNH THƯỜNG (NORMAL)**")
                st.metric(label="Xác suất Bình thường", value=f"{normal_pct:.2f}%")
                st.progress(int(normal_pct))
                st.info("ℹ️ Phổi sáng, chưa phát hiện dấu hiệu nguy hiểm vượt ngưỡng cài đặt.")
                
            st.markdown("---")
            with st.expander("Xem chi tiết kỹ thuật"):
                st.write(f"- Xác suất Bình thường (Normal): **{normal_pct:.2f}%**")
                st.write(f"- Xác suất Viêm phổi (Pneumonia): **{pneumonia_pct:.2f}%**")
                st.write(f"- Ngưỡng cài đặt (Threshold): **{threshold}**")
                if pneumonia_prob_val >= threshold:
                    st.write("👉 **Kết luận: PNEUMONIA** (Vì Xác suất Viêm phổi >= Ngưỡng)")
                else:
                    st.write("👉 **Kết luận: NORMAL** (Vì Xác suất Viêm phổi < Ngưỡng)")

# --- Footer ---
st.markdown("---")
st.caption("Developed by Group [Tên Nhóm] - Data Mining Course.")