import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize

# استيراد الكلاسات الخاصة بك
from model_stage3 import StressRecognitionModel
from dataset_stage3 import StressDataset

# --- الإعدادات (تصحيح مسار الملف) ---
# تم حذف حرف h الزائد من البداية لتجنب FileNotFoundError
MODEL_PATH = "/home/hajr/stress/stage_3_improve/stage_3_results/stress_multi_task_rppg_v2/final_gold_model/final_gold_stress_model.pth"
DATA_PATH = "/home/hajr/stress/processed_data_final_v5_full"
SAVE_DIR = "./visual_analysis_improve"
os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_visuals():
    # 1. تحميل الموديل
    print("🚀 Loading Gold Model...")
    model = StressRecognitionModel().to(DEVICE)
    
    # تحميل الأوزان (استخدام weights_only=True لتجنب التحذيرات المستقبلية)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return

    model.eval()

    # 2. تحميل بيانات الاختبار (أول 8 أشخاص)
    all_files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith('.npy')])
    subjects = sorted(list(set([f.split('_')[0] for f in all_files])))
    test_subs = subjects[:8]
    print(f"📦 Loading data for subjects: {test_subs}")
    test_loader = DataLoader(StressDataset(DATA_PATH, test_subs), batch_size=1, shuffle=False)

    all_labels = []
    all_preds = []
    all_probs = [] # سنحتاج الاحتمالات لرسم ROC Curve بدقة
    all_features = []

    print("📊 Extracting predictions and features...")
    with torch.no_grad():
        for batch in test_loader:
            video = batch['video'].to(DEVICE)
            labels = batch['stress_level'].to(DEVICE)
            
            outputs = model(video)
            
            # استخراج الميزات (Features) بنفس الطريقة الأصلية
            features = model.physio_extractor.encoder.extract_features(video)[3]
            refined = model.physio_extractor.attn_stacks[3](features)
            final_feat = model.physio_extractor.long_range_refinement(refined)
            avg_p = torch.mean(final_feat, dim=1)
            max_p, _ = torch.max(final_feat, dim=1)
            combined_feat = torch.cat([avg_p, max_p], dim=1)

            # تخزين النتائج
            probs = torch.softmax(outputs['level'], dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(torch.argmax(outputs['level'], dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_features.extend(combined_feat.cpu().numpy())

    # تحويل القوائم إلى مصفوفات Numpy
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_features = np.array(all_features)
    labels_names = ['Rest', 'Low Stress', 'High Stress']

    # --- 1. Confusion Matrix ---
    print("🎨 Plotting Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_names, yticklabels=labels_names)
    plt.title('Confusion Matrix: Stress Level Recognition')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()

    # --- 2. t-SNE (توزيع الميزات) ---
    print("🎨 Plotting t-SNE Feature Distribution...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(all_features)
    
    plt.figure(figsize=(10, 7))
    colors = ['#1f77b4', '#ff7f0e', '#d62728'] 
    for i in range(3):
        indices = np.where(all_labels == i)
        plt.scatter(features_2d[indices, 0], features_2d[indices, 1], 
                    c=colors[i], label=labels_names[i], alpha=0.6, edgecolors='w')
    plt.legend()
    plt.title('t-SNE Visualization of MS-CAM-Net Learned Features')
    plt.savefig(os.path.join(SAVE_DIR, "tsne_features.png"), dpi=300)
    plt.close()

    # --- 3. المعايير الجديدة: Cohen's Kappa & Classification Report ---
    print("📝 Calculating Advanced Metrics...")
    kappa = cohen_kappa_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=labels_names)
    
    with open(os.path.join(SAVE_DIR, "advanced_metrics_report.txt"), "w") as f:
        f.write("=== Advanced Statistical Analysis ===\n")
        f.write(f"Cohen’s Kappa Coefficient: {kappa:.4f}\n")
        f.write("(Interpretation: >0.61 Substantial, >0.81 Almost Perfect)\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report)
    print(f"✅ Kappa Score: {kappa:.3f}")

    # --- 4. المعايير الجديدة: ROC Curve & AUC ---
    print("📈 Plotting ROC Curves...")
    y_test_bin = label_binarize(all_labels, classes=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{labels_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve Analysis')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(SAVE_DIR, "roc_curves.png"), dpi=300)
    plt.close()

    # --- 5. المعايير الجديدة: Sensitivity (Recall) & Precision Bar Chart ---
    print("📊 Plotting Sensitivity Analysis...")
    metrics_dict = classification_report(all_labels, all_preds, output_dict=True)
    recalls = [metrics_dict[str(i)]['recall'] for i in range(3)]
    precisions = [metrics_dict[str(i)]['precision'] for i in range(3)]

    x = np.arange(len(labels_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width/2, recalls, width, label='Recall (Sensitivity)', color='#2ca02c')
    ax.bar(x + width/2, precisions, width, label='Precision', color='#9467bd')
    
    ax.set_ylabel('Score Value')
    ax.set_title('Per-Class Performance Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_names)
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.savefig(os.path.join(SAVE_DIR, "performance_breakdown.png"), dpi=300)
    plt.close()

    print(f"\n✨ Process Complete! All visuals and reports saved in: {SAVE_DIR}")

if __name__ == "__main__":
    generate_visuals()
