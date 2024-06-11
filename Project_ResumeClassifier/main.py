import nltk
import pickle
import re
import streamlit as st

nltk.download('punkt')
nltk.download('stopwords')

## load models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))



def cleanResume(Text):
  # replace url that starts with http followed by any non-whitespace characters (\S+), and then a whitespace character (\s) with space
  clean_Text = re.sub('https\S+\s', ' ', Text)
  #replace RT (retweets for Twitter) with space. Ensure "RT" and "cc" are whole words
  clean_Text = re.sub(r'\bRT\b|\bcc\b', ' ', clean_Text)
  #repace hashtag with space
  clean_Text = re.sub('#\S+\s', ' ', clean_Text)
  #replace mentions and emails with space
  clean_Text = re.sub('@\S+\s', ' ', clean_Text)
  #replace special character
  clean_Text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_Text)
  #remove non ASCII characters. NOTE: r'' before a string in a regular expression, it means that the string inside
  # is treated exactly as it appears, without interpreting any backslashes as escape sequences.
  clean_Text = re.sub(r'[^\x00-\x7f]', ' ', clean_Text)
  #Replace multiple spaces with a single space
  clean_Text = re.sub('\s+', ' ', clean_Text)

  return clean_Text


### Making the web app
def main():
    st.title('Resume Screening App')
  
    uploaded_file = st.file_uploader('Upload Resume', type = ['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            resume_byte = uploaded_file.read()
            resume_text = resume_byte.decode('utf-8')
        except:
            resume_text = resume_byte.decode('latin-1')

        cleaned_resume = cleanResume(resume_text)
        input_features = tfidf.transform([cleaned_resume])
        prediction = clf.predict(input_features)[0]

        st.write(prediction)
        
        category_mapping = {
            0: 'Advocate',
            1: 'Arts',
            2: 'Automation Testing',
            3: 'Blockchain',
            4: 'Business Analyst',
            5: 'Civil Engineer',
            6: 'Data Science',
            7: 'Database',
            8: 'DevOps Engineer',
            9: 'DotNet Developer',
            10: 'ETL Developer',
            11: 'Electrical Engineering',
            12: 'HR',
            13: 'Hadoop',
            14: 'Health and fitness',
            15: 'Java Developer',
            16: 'Mechanical Engineer',
            17: 'Network Security Engineer',
            18: 'Operations Manager',
            19: 'PMO',
            20: 'Python Developer',
            21: 'SAP Developer',
            22: 'Sales',
            23: 'Testing',
            24: 'Web Designing'
        }

        category_name = category_mapping.get(prediction, 'Unknown')
        st.write('Predicted category:', category_name)


if __name__ == "__main__":
  main()