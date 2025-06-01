import streamlit as st
from datetime import datetime
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Personal Expense Tracker", layout="wide")
st.title("💰 Personal Expense Tracker")

if "show_add_expense_form" not in st.session_state:
    st.session_state.show_add_expense_form = False

if "view_exp" not in st.session_state:
    st.session_state.view_exp = False


class ExpenseTracker:
    def __init__(self):
        self.file_name = "Expenses.csv"

    def add_expenses(self):
        st.header("➕ Add Expense")

        category = st.text_input("Enter the category").strip().upper()
        amount_input = st.text_input("Enter the amount")
        date = st.date_input("Enter the date")

        if st.button("Add"):
            if not category:
                st.error("Category is required")
            elif not amount_input:
                st.error("Amount is required")
            else:
                try:
                    amount = float(amount_input)
                except ValueError:
                    st.error("Amount must be a number")
                    return

                file_exists = os.path.exists(self.file_name) and os.path.getsize(self.file_name) > 0

                with open(self.file_name, mode='a', newline="") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['Category', 'Amount', 'Date'])
                    writer.writerow([category, amount, date])

                st.success("Expense added successfully ✅")
                st.session_state.show_add_expense_form = False

    def view_expenses(self):
        st.header("📋 View Expenses")

        try:
            df = pd.read_csv(self.file_name)
        except FileNotFoundError:
            st.warning("No expenses found yet. Add some first.")
            return

        categories = df['Category'].unique()
        selected_option = st.selectbox("Select a category", categories)
        filtered = df[df['Category'] == selected_option]
        total_amount = filtered['Amount'].sum()

        st.write(f"**Total amount spent on {selected_option}: ₹{total_amount}**")
        st.dataframe(filtered)

    def data_analysis(self):
        st.header("📊 Spending Analysis")

        try:
            df = pd.read_csv(self.file_name)
        except FileNotFoundError:
            st.warning("No data to analyze. Please add expenses first.")
            return

        bar_data = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Bar Chart")
            plt.bar(bar_data.index, bar_data.values, color='skyblue')
            plt.xticks(rotation=45)
            plt.ylabel("Amount")
            plt.title("Amount Spent by Category")
            st.pyplot(plt)
            plt.clf()

        with col2:
            st.subheader("Pie Chart")
            plt.pie(bar_data, labels=bar_data.index, autopct="%1.1f%%", startangle=140)
            plt.axis('equal')
            plt.title("Spending Distribution")
            st.pyplot(plt)
            plt.clf()


tracker = ExpenseTracker()
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("➕ Add Expenses"):
        st.session_state.show_add_expense_form = True
        st.session_state.view_exp = False
        st.session_state.show_analysis = False

with col2:
    if st.button("📄 View Expenses"):
        st.session_state.view_exp = True
        st.session_state.show_add_expense_form = False
        st.session_state.show_analysis = False

with col3:
    if st.button("📊 Show Spent Analysis"):
        st.session_state.show_analysis = True
        st.session_state.view_exp = False
        st.session_state.show_add_expense_form = False

        
if st.session_state.show_add_expense_form:
    tracker.add_expenses()
elif st.session_state.view_exp:
    tracker.view_expenses()
elif st.session_state.show_analysis:
    tracker.data_analysis()