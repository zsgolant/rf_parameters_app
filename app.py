import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Streamlit UI
def main():
    st.title("Random Forest Classifier")

    # File upload
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Display dataframe
        st.write("Data Preview:")
        st.write(df.head())

        # Select y column
        y_column = st.selectbox("Select the target column (y)", options=df.columns)

        if st.button("Fit Model"):
            # Split data into X and y
            X = df.drop(columns=[y_column])
            y = df[y_column]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Fit Random Forest Classifier
            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train, y_train)

            # Prediction and evaluation
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Display results
            st.subheader("Model Evaluation")
            st.write(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()

