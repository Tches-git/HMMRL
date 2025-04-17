from src.graph_builder import GraphBuilder

def main():
    bert_model = None
    bert_tokenizer = None
    
    builder = GraphBuilder()
    
    image_paths = [
        r"C:\Users\28489\Desktop\paired\31\1.jpg",
        r"C:\Users\28489\Desktop\paired\31\2.jpg",
        r"C:\Users\28489\Desktop\paired\31\4.jpg",
        r"C:\Users\28489\Desktop\paired\31\5.jpg",
        r"C:\Users\28489\Desktop\paired\31\6.jpg"
    ]
    texts = ["图像1描述", "图像2描述", "图像4描述", "图像5描述", "图像6描述"]
    
    try:
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