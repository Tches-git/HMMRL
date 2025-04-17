import tkinter as tk
from tkinter import filedialog
from src.graph_builder import GraphBuilder
import torch

def select_image_files():
    """使用文件对话框让用户选择多张图片"""
    root = tk.Tk()
    root.withdraw()
    filetypes = [("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
    files = filedialog.askopenfilenames(
        title="选择图像文件",
        filetypes=filetypes,
        initialdir="C:\\Users\\28489\\Desktop\\paired\\31"
    )
    root.destroy()
    return list(files)

def main():
    # 检测 GPU 可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    bert_model = None
    bert_tokenizer = None
    
    # 传递设备参数
    builder = GraphBuilder(device=device)
    
    # 让用户选择图像文件
    image_paths = select_image_files()
    if not image_paths:
        print("未选择任何图像文件，程序退出。")
        return
    
    # 根据选择的文件数量生成默认文本描述
    texts = [f"图像{i+1}描述" for i in range(len(image_paths))]
    
    try:
        print("选择的图像文件：")
        for path in image_paths:
            print(f"- {path}")
        
        graph, stitched_image = builder.build_from_images_with_rl(
            image_paths=image_paths,
            texts=texts,
            bert_model=bert_model,
            bert_tokenizer=bert_tokenizer,
            max_iterations=10,
            epsilon=0.1
        )
        
        print(f"构建的图包含 {len(graph.nodes)} 个节点和 {len(graph.edges)} 条边")
        graph.visualize(stitched_image)
        
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")

if __name__ == "__main__":
    main()