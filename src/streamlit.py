import streamlit as st
import pandas as pd
import pyprojroot
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

root = pyprojroot.find_root(pyprojroot.has_dir(".git"))

# Set page title, icon, and layout
st.set_page_config(
    page_title="Supervise-me",
    page_icon=":mouse:",
    layout="wide"
)

# Set page header and title
st.subheader("Hi, we are supervise-me :wave:")
st.write("---")


def main():
    # Load the dataset
    df = pd.read_csv(root / 'data' / 'processed' / 'meta.csv')   

    st.title('Dataset Overview')
    st.write(df.head())

    # Show column names
    st.sidebar.subheader('Column Names')
    column_names_df = pd.DataFrame({'Columns': df.columns})
    st.sidebar.write(column_names_df)

    
    st.write("---")
    # Get all combinations of columns
    selected_number = st.sidebar.selectbox('Select a number', [2, 3])
    selected_columns = st.sidebar.multiselect('Select columns for combinations', df.columns[1:])
    combinations = list(itertools.combinations(selected_columns, selected_number))

    #Count and display combinations
    for combination in combinations:
        count = df.groupby(list(combination)).size().reset_index(name='Count')
        st.subheader(f'Combination: {", ".join(combination)}')
        st.write(count)  

        if len(selected_columns) == 2:
            # Plot bar chart for combination
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=count, x=combination[0], y='Count', hue=combination[1])
            ax.set_xlabel(combination[0])
            ax.set_ylabel("Count")
            ax.set_title(f'Bar Chart: {", ".join(combination)}, and count')
            ax.legend(title=combination[1])
            st.pyplot(fig)
        

if __name__ == "__main__":
    main()
