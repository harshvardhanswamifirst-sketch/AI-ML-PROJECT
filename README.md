🚀 Fake Review Detector

A simple Machine Learning project that detects whether a product review is Fake or Genuine using Natural Language Processing (NLP).

📌 Project Overview

Online reviews play a major role in customer decisions, but many are fake or misleading. This project uses Machine Learning to automatically classify reviews and help users identify trustworthy feedback.

🎯 Features Detects fake vs genuine reviews Uses NLP techniques (TF-IDF) Logistic Regression model Real-time user input prediction Simple and beginner-friendly

🛠️ Tech Stack Python Pandas NumPy Scikit-learn Regex (re)

⚙️ How It Works Reviews dataset is created (fake & real) Text is cleaned and preprocessed TF-IDF converts text → numerical data Logistic Regression model is trained User enters a review → prediction is shown

📂 Project Structure fake-review-detector/ │ ├── fake_review_detector.py ├── README.md └── Project_Report.pdf

💡 Example Output Enter a review: This product is amazing must buy!!! Result: ⚠️ Fake Review

Enter a review: The product works fine but quality is average Result: ✅ Genuine Review

📊 Model Performance Algorithm: Logistic Regression Accuracy: ~85% - 95% Evaluation Metrics: Precision, Recall, F1-score

🚧 Limitations Uses small synthetic dataset May not perform well on real-world data Limited training examples

🔮 Future Improvements Use real datasets (Amazon/Flipkart reviews) Build web app using Flask Add Deep Learning (LSTM/BERT) Improve dataset size and accuracy 🤝 Contributing

Feel free to fork this repo and improve the project!

📜 License

This project is for educational purposes only.

👨‍💻 Author

Harshvardhan swami

REG.NO.- 25BCE11122

⭐ If you like this project

Give it a ⭐ on GitHub!
