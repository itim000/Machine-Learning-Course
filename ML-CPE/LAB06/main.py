from data_loader import load_heart_data
from preprocessing import prepare_data
from nn_model import train_model
from evaluate import evaluate_and_save, plot_prediction_samples  # เพิ่มตรงนี้

def main():
    print("1. Loading Heart Disease Dataset...")
    X, y = load_heart_data('heart.csv')
    
    print("2. Preprocessing & Scaling Data...")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(X, y)
    
    print("3. Training Neural Network...")
    model, history = train_model(
        X_train, y_train, X_val, y_val, output_dir="outputs", epochs=50, batch_size=16
    )
    
    print("4. Evaluating & Saving Visualizations...")
    acc = evaluate_and_save(model, X_test, y_test, history, output_dir='outputs')
    
    # เพิ่มบรรทัดนี้เพื่อสร้างไฟล์ prediction_sample.png
    plot_prediction_samples(model, X_test, y_test, output_dir='outputs', num_samples=4)
    
    print(f"\nDone! Final Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()