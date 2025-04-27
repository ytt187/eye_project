# -*- coding: utf-8 -*-
"""
大赛专用眼科AI系统 - 诊断建议增强版
"""
import gradio as gr
from pyngrok import ngrok
import threading
import webbrowser
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import socket
from train_eye import MedicalDataset, AttEfficientNet, MedicalAugmentation

class ContestSystem:
    def __init__(self, model_path, class_names):
        self.annotation_colors = {
            "青光眼": (255, 0, 0),
            "糖尿病视网膜病变": (0, 255, 0),
            "黄斑疤痕": (0, 0, 255),
            "default": (255, 255, 0)
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AttEfficientNet(num_classes=len(class_names)).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.class_names = class_names
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.setup_ui()

    def predict(self, img_pil):
        try:
            img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            self.model.save_features = True
            
            with torch.no_grad():
                logits = self.model(img_tensor)
                probs = F.softmax(logits, dim=1)
            
            heatmap = self.generate_heatmap(img_pil)
            pred_dict = {self.class_names[i]: float(probs[0][i]) 
                        for i in range(len(self.class_names))}
            
            # 获取最高概率疾病
            top_disease = max(pred_dict, key=pred_dict.get)
            return top_disease, heatmap, self.generate_clinical_advice(pred_dict)
        except Exception as e:
            print(f"预测错误: {e}")
            return "诊断错误", np.zeros((224,224,3)), "无法生成建议"
        
    def generate_heatmap(self, img_pil):
        feature_maps = self.model.feature_maps[0].cpu().numpy()
        weights = np.mean(feature_maps, axis=(1, 2))
        
        cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * feature_maps[i]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, img_pil.size)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        heatmap = plt.get_cmap('jet')(cam)[:, :, :3] * 255
        return (0.5 * np.array(img_pil) + 0.5 * heatmap).astype(np.uint8)

    
    def generate_clinical_advice(self, pred_dict):
        # 获取前三疾病及其概率
        sorted_diseases = sorted(pred_dict.items(), key=lambda x:x[1], reverse=True)[:3]
        top_disease, top_prob = sorted_diseases[0]
        
        # 临床建议数据库
        advice_db = {
            "糖尿病视网膜病变": [
                "【主要建议】",
                "1. 立即进行糖化血红蛋白检测（目标值≤7%）",
                "2. 3个月内复查眼底荧光造影",
                "3. 根据分期考虑抗VEGF治疗",
                "4. 建议内分泌科联合诊疗"
            ],
            "青光眼": [
                "【紧急处理】",
                "1. 24小时眼压监测（每日8/12/16/20时测量）",
                "2. 视野检查和OCT视神经分析",
                "3. 首选前列腺素类滴眼液降眼压",
                "4. 眼压控制目标：≤21mmHg"
            ],
            "黄斑疤痕": [
                "【诊疗方案】",
                "1. OCT检查明确疤痕活动性",
                "2. 每月视力监测（使用标准视力表）",
                "3. 考虑玻璃体腔注射雷珠单抗",
                "4. 避免剧烈运动防止出血"
            ],
            "正常": [
                "【健康指导】",
                "1. 每年一次全面眼底检查",
                "2. 保持血糖<6.1mmol/L，血压<140/90mmHg",
                "3. 每日用眼不超过8小时，每小时休息5分钟",
                "4. 建议补充叶黄素和omega-3"
            ],
            "高血压视网膜病变": [
                "【管理要点】",
                "1. 24小时动态血压监测",
                "2. 每月眼底照相跟踪动脉变化",
                "3. 血压控制目标：<130/80mmHg",
                "4. 建议心内科联合会诊"
            ]
        }

        # 构建建议文本
        advice = "🩺 预测可能性前三疾病：\n"
        for i, (disease, prob) in enumerate(sorted_diseases):
            advice += f"{i+1}. {disease}（可能性：{prob*100:.1f}%）\n"
        
        advice += "\n📋 临床决策建议：\n"
        advice += "\n".join(advice_db.get(top_disease, 
            ["⚠️ 未匹配到明确诊疗方案，建议：", 
             "1. 进行OCT、FFA等专项检查",
             "2. 三甲医院眼科专家会诊",
             "3. 2周后复查对比变化"]))
        
        return advice
    def generate_annotations(self, img_pil, pred_class, cam, threshold=0.3):
        img_np = np.array(img_pil)
        color = self.annotation_colors.get(pred_class, self.annotation_colors["default"])

        binary_mask = (cam > threshold).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        gradio_annotations = []
        features = []
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) > 50:
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

                mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 1, thickness=cv2.FILLED)

                gradio_annotations.append((
                    mask,
                    f"{pred_class}_{i + 1}: {int(area)}px"
                ))
                features.append([
                    f"区域_{i + 1}",
                    int(area),
                    f"{circularity:.2f}",
                    f"{w}x{h}",
                    f"({x + int(w / 2)},{y + int(h / 2)})"
                ])

        return Image.fromarray(img_np), gradio_annotations, features
    
    
    def full_analysis(self, img_pil, threshold=0.3):
        try:
            img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            self.model.save_features = True
            
            with torch.no_grad():
                logits = self.model(img_tensor)
                probs = F.softmax(logits, dim=1)
            
            pred_dict = {self.class_names[i]: float(probs[0][i]) 
                        for i in range(len(self.class_names))}
            top_disease = max(pred_dict, key=pred_dict.get)
            
            # Generate CAM
            feature_maps = self.model.feature_maps[0].cpu().numpy()
            weights = np.mean(feature_maps, axis=(1, 2))
            cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * feature_maps[i]
            
            cam = cv2.resize(cam, img_pil.size)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            # Generate annotations
            annotated_img, annotations, features = self.generate_annotations(
                img_pil, top_disease, cam, threshold
            )
            
            return (
                top_disease,  # 诊断结论（字符串）
                self.generate_heatmap(img_pil),  # 热力图（numpy数组）
                (np.array(img_pil), annotations),  # 标注图像和注释
                features,  # 特征数据（列表）
                self.generate_clinical_advice(pred_dict)  # 临床建议（字符串）
            )
        except Exception as e:
            print(f"完整分析错误: {e}")
            # 返回与正常结构一致的占位数据
            placeholder_img = np.zeros((224, 224, 3), dtype=np.uint8)
            return (
                "分析失败",
                placeholder_img,
                (placeholder_img, []),
                [],
                "分析失败，请联系技术支持"
            )

    def setup_ui(self):
        with gr.Blocks(
            title="智能眼科诊疗系统",
            theme=gr.themes.Soft(primary_hue="blue"),
            css="""
            .diagnosis-panel { padding: 15px; border-radius: 8px; border: 1px solid #eee; }
            .analysis-btn { margin-top: 15px; width: 200px; }
            .advice-box { font-family: 'Microsoft YaHei'; line-height: 1.8; }
            .example-label { 
                text-align: center; 
                font-weight: bold; 
                margin-top: 5px; 
                color: #333;
            }
            .example-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                margin-bottom: 15px;
            }
            """
        ) as demo:
            gr.Markdown("# 视网膜疾病智能诊断平台")
        
            with gr.TabItem("疾病诊断", id="diagnosis"):
                with gr.Row():
                    # 左侧上传区
                    with gr.Column(scale=1):
                        gr.Markdown("## 上传眼底照片")
                        img_input = gr.Image(type="pil", height=400)
                        
                        # 修改后的典型病例库 - 展示10个类别的示例图像
                        gr.Markdown("### 典型病例库")
                        with gr.Row():
                            with gr.Column(min_width=120):
                                gr.Markdown("**糖尿病视网膜病变**")
                                gr.Examples(
                                    examples=["examples/DR9.jpg"],
                                    inputs=img_input,
                                    label=None,
                                    examples_per_page=1,
                                   
                                )
                            with gr.Column(min_width=120):
                                gr.Markdown("**青光眼**")
                                gr.Examples(
                                    examples=["examples/glaucoma_sample.jpg"],
                                    inputs=img_input,
                                    label=None,
                                    examples_per_page=1,
                                   
                                )
                            with gr.Column(min_width=120):
                                gr.Markdown("**黄斑疤痕**")
                                gr.Examples(
                                    examples=["examples/Macular Scar1.jpg"],
                                    inputs=img_input,
                                    label=None,
                                    examples_per_page=1,
                                    
                                )
                        '''
                        with gr.Row():
                            with gr.Column(min_width=120):
                                gr.Markdown("**CSCR**")
                                gr.Examples(
                                    examples=["examples/CSCR3.jpg"],
                                    inputs=img_input,
                                    label=None,
                                    examples_per_page=1,
                                    
                                )
                            with gr.Column(min_width=120):
                                gr.Markdown("**近视**")
                                gr.Examples(
                                    examples=["examples/Myopia7.jpg"],
                                    inputs=img_input,
                                    label=None,
                                    examples_per_page=1,
                                  
                                    
                                )
                            with gr.Column(min_width=120):
                                gr.Markdown("**翼状絮肉**")
                                gr.Examples(
                                    examples=["examples/Pterygium3.jpg"],
                                    inputs=img_input,
                                    label=None,
                                    examples_per_page=1,
                                    
                                )
                        
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**视网膜脱离**")
                                gr.Examples(
                                    examples=["examples/Retinal Detachment1.jpg"],
                                    inputs=img_input,
                                    label=None,
                                    examples_per_page=1,
                                   
                                )
                            with gr.Column():
                                gr.Markdown("**RP**")
                                gr.Examples(
                                    examples=["examples/Retinitis Pigmentosa1.jpg"],
                                    inputs=img_input,
                                    label=None,
                                    examples_per_page=1,
                                  
                                )
                            with gr.Column():
                                gr.Markdown("**椎间盘水肿**")
                                gr.Examples(
                                    examples=["examples/Disc Edema1.jpg"],
                                    inputs=img_input,
                                    label=None,
                                    examples_per_page=1,
                                   
                                )
                                '''

                    # 右侧主显示区
                    with gr.Column(scale=2):
                        # 基础诊断面板
                        with gr.Column(visible=True) as basic_panel:
                            gr.Markdown("## AI诊断结果")
                            diagnosis = gr.Textbox(label="初步诊断结论", interactive=False)
                            
                            # 热力图与按钮
                            with gr.Column():
                                heatmap = gr.Image(label="病变热力图", height=300)
                                analysis_btn = gr.Button(
                                    "显示详细分析 ▶", 
                                    elem_classes="analysis-btn",
                                    variant="secondary"
                                )
                            
                            with gr.Accordion("临床决策建议", open=True):
                                advice = gr.Textbox(lines=8, interactive=False, 
                                                  elem_classes="advice-box")

                        # 详细分析面板（默认隐藏）
                        with gr.Column(visible=False) as detail_panel:
                            with gr.Row():
                                back_btn = gr.Button("◀ 返回概览", variant="secondary")
                                gr.Markdown("## 图像特征分析")
                            
                            # 标注图例
                            with gr.Row(elem_classes="annotation-legend"):
                                for name, color in self.annotation_colors.items():
                                    gr.HTML(f"""
                                    <div class='legend-item'>
                                        <div class='legend-color' style='background:rgb{color};'></div>
                                        <span>{name}</span>
                                    </div>
                                    """)
                            
                            # 分析控件
                            with gr.Row():
                                threshold_slider = gr.Slider(
                                    0.1, 0.9, 0.3, 
                                    label="标注敏感度阈值",
                                    step=0.05
                                )
                                refresh_btn = gr.Button("重新生成标注", variant="secondary")
                            
                            # 分析结果展示
                            analysis_output = gr.AnnotatedImage(
                                label="病变区域标注",
                                show_legend=False,
                                height=400
                            )
                            feature_table = gr.Dataframe(
                                headers=["区域", "面积(px)", "圆形度", "尺寸", "位置"],
                                datatype=["str", "number", "str", "str", "str"],
                                interactive=False
                            )

                # 事件绑定
                img_input.change(
                    fn=self.full_analysis,
                    inputs=[img_input, threshold_slider],
                    outputs=[diagnosis, heatmap, analysis_output, feature_table, advice]
                )
                
                analysis_btn.click(
                    lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(value="隐藏分析 ▼")],
                    outputs=[basic_panel, detail_panel, analysis_btn]
                )

                back_btn.click(
                    lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(value="显示详细分析 ▶")],
                    outputs=[basic_panel, detail_panel, analysis_btn]
                )

                threshold_slider.change(
                    lambda img, t: self.full_analysis(img, t)[2:4],
                    inputs=[img_input, threshold_slider],
                    outputs=[analysis_output, feature_table]
                )

                refresh_btn.click(
                    lambda img: self.full_analysis(img, 0.3)[2:4],
                    inputs=img_input,
                    outputs=[analysis_output, feature_table]
                )

            # ===== 健康科普标签页 =====
            with gr.TabItem("健康科普", id="education"):
                gr.Markdown("## 眼病科普知识")
                with gr.Tabs():
                    # ----------------- 糖尿病视网膜病变 -----------------
                    with gr.TabItem("糖尿病视网膜病变"):
                        with gr.Row():
                            # 左侧图像可视化
                            with gr.Row():
                                with gr.Column(scale=1):
                                    gr.Markdown("### 眼底病变特征图示")
                                    gr.Image("examples/DR9.jpg", height=300)
                                    gr.Markdown("*图：微血管瘤（红色箭头）和出血斑*", elem_classes="caption")
                                            
                
                            # 右侧疾病描述
                            with gr.Column(scale=1):
                                gr.Markdown("### 疾病概述")
                                gr.Markdown("""
                                **📌 核心特征**  
                                - 微血管瘤形成  
                                - 视网膜出血、渗出  
                                - 黄斑水肿（视力下降主因）  

                                **❗ 高危人群**  
                                - 糖尿病病程>10年  
                                - 血糖控制不佳（HbA1c>7%）  

                                **🩺 筛查建议**  
                                - 1型糖尿病：确诊5年后每年检查  
                                - 2型糖尿病：确诊即开始每年检查  
                                """, elem_classes="disease-desc")

                    # ----------------- 青光眼 -----------------
                    with gr.TabItem("青光眼"):
                        with gr.Row():
                            # 左侧图像可视化
                            with gr.Column(scale=1):
                                gr.Markdown("### 视神经损伤图示")
                                gr.Image("examples/glaucoma_sample.jpg", height=300)
                                gr.Markdown("*图：视杯扩大（C/D>0.7）*", elem_classes="caption")
                            
                            # 右侧疾病描述
                            with gr.Column(scale=1):
                                gr.Markdown("### 疾病概述")
                                gr.Markdown("""
                                **📌 核心特征**  
                                - 进行性视神经损伤  
                                - 特征性视野缺损（鼻侧阶梯）  
                                - 眼压>21mmHg（部分患者可正常）  

                                **❗ 高危人群**  
                                - 直系亲属有青光眼病史  
                                - 高度近视（>600度）  

                                **🩺 筛查建议**  
                                - 40岁以上：每2年检查眼压和视神经  
                                - 高危人群：每年OCT检查  
                                """, elem_classes="disease-desc")
                    # ----------------- 黄斑疤痕 -----------------
                    with gr.TabItem("黄斑疤痕"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### OCT影像图示")
                                gr.Image("examples/Macular Scar1.jpg", height=300)
                                gr.Markdown("*图：黄斑区纤维增生（白色箭头）*", elem_classes="caption")
                            with gr.Column(scale=1):
                                gr.Markdown("### 疾病概述")
                                gr.Markdown("""
                                **📌 核心特征**  
                                - 中心视力不可逆损伤  
                                - 视物变形  
                                **❗ 常见病因**  
                                - 病理性近视  
                                - 外伤后遗症  
                                **🩺 治疗建议**  
                                - 抗VEGF药物注射  
                                """)
                                
                   # ----------------- 近视性视网膜病变 -----------------
                    with gr.TabItem("近视病变"):
                       with gr.Row():
                           with gr.Column(scale=1):
                               gr.Markdown("### 高度近视眼底")
                               gr.Image("examples/Myopia1.jpg", height=300)
                               gr.Markdown("*图：视网膜变薄（豹纹状改变）*", elem_classes="caption")
                           with gr.Column(scale=1):
                               gr.Markdown("### 疾病概述")
                               gr.Markdown("""
                               **📌 核心特征**  
                               - 度数>600度  
                               - 视网膜萎缩灶  
                               **❗ 并发症风险**  
                               - 视网膜脱离风险增加8倍  
                               **🩺 防控建议**  
                               - 每年散瞳查眼底  
                               """)

                   # ----------------- 视网膜色素变性 -----------------
                    with gr.TabItem("视网膜色素变性（RP）"):
                       with gr.Row():
                           with gr.Column(scale=1):
                               gr.Markdown("### 眼底表现")
                               gr.Image("examples/Retinitis Pigmentosa1.jpg", height=300)
                               gr.Markdown("*图：骨细胞样色素沉着*", elem_classes="caption")
                           with gr.Column(scale=1):
                               gr.Markdown("### 疾病概述")
                               gr.Markdown("""
                               **📌 核心特征**  
                               - 夜盲为首发症状  
                               - 进行性视野缩小  
                               **❗ 遗传特点**  
                               - 50%为常染色体隐性遗传  
                               **🩺 治疗进展**  
                               - 基因治疗临床试验中  
                               """)

                   # ----------------- 视网膜脱离 -----------------
                    with gr.TabItem("视网膜脱离"):
                       with gr.Row():
                           with gr.Column(scale=1):
                               gr.Markdown("### 超声影像图")
                               gr.Image("examples/Retinal Detachment1.jpg", height=300)
                               gr.Markdown("*图：视网膜隆起（红色箭头）*", elem_classes="caption")
                           with gr.Column(scale=1):
                               gr.Markdown("### 疾病概述")
                               gr.Markdown("""
                               **📌 核心特征**  
                               - 眼前黑影遮挡  
                               - 闪光感先兆  
                               **❗ 紧急处理**  
                               - 需24小时内手术  
                               **🩺 高危人群**  
                               - 高度近视、眼外伤史  
                               """)

                   # ----------------- 翼状胬肉 -----------------
                    with gr.TabItem("翼状胬肉"):
                       with gr.Row():
                           with gr.Column(scale=1):
                               gr.Markdown("### 眼前节照片")
                               gr.Image("examples/Pterygium3.jpg", height=300)
                               gr.Markdown("*图：鼻侧结膜增生*", elem_classes="caption")
                           with gr.Column(scale=1):
                               gr.Markdown("### 疾病概述")
                               gr.Markdown("""
                               **📌 核心特征**  
                               - 睑裂区结膜增生  
                               - 可侵入角膜  
                               **❗ 诱发因素**  
                               - 长期紫外线暴露  
                               **🩺 治疗原则**  
                               - 影响视力时手术切除  
                               """)

                   # ----------------- 中浆（CSCR） -----------------
                    with gr.TabItem("中浆（CSCR）"):
                       with gr.Row():
                           with gr.Column(scale=1):
                               gr.Markdown("### OCT影像")
                               gr.Image("examples/CSCR3.jpg", height=300)
                               gr.Markdown("*图：神经上皮层脱离*", elem_classes="caption")
                           with gr.Column(scale=1):
                               gr.Markdown("### 疾病概述")
                               gr.Markdown("""
                               **📌 核心特征**  
                               - 中心视力突然下降  
                               - 视物变形  
                               **❗ 好发人群**  
                               - 30-50岁男性  
                               - A型性格  
                               **🩺 自愈倾向**  
                               - 80%患者3-6个月自愈  
                               """)

                   # ----------------- 视盘水肿 -----------------
                    with gr.TabItem("视盘水肿"):
                       with gr.Row():
                           with gr.Column(scale=1):
                               gr.Markdown("### 眼底彩照")
                               gr.Image("examples/Disc Edema1.jpg", height=300)
                               gr.Markdown("*图：视盘边界模糊*", elem_classes="caption")
                           with gr.Column(scale=1):
                               gr.Markdown("### 疾病概述")
                               gr.Markdown("""
                               **📌 核心特征**  
                               - 视盘隆起>3D  
                               - 静脉搏动消失  
                               **❗ 病因警示**  
                               - 需排查颅内压增高  
                               **🩺 紧急处理**  
                               - 需颅脑MRI检查  
                               """)

               # CSS样式（修正版）
                css = """
               .caption { font-size:0.8em; text-align:center; color:#666; margin-top:-10px; }
               .disease-desc { background:#f8f9fa; padding:15px; border-radius:8px; }
               .disease-desc ul { padding-left:20px; }
               .tab { min-width:120px !important; }
               """
                demo.css = css
         

            self.demo = demo
    
    

    def launch(self, port=7860):
        """直接使用Gradio原生启动"""
        self.demo.launch(server_name="0.0.0.0", server_port=port)

    

if __name__ == "__main__":
    data_dir = "Augmented Dataset"
    transform = MedicalAugmentation()
    train_set = MedicalDataset(data_dir, transform, 'train')
    CLASS_NAMES = train_set.classes
    MODEL_PATH = "newbest_model.pth"
    # 禁用GPU（云环境可能无GPU）
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    system = ContestSystem(MODEL_PATH, CLASS_NAMES)
    system.launch()
