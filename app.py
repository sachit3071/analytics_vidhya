import requests
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import os
import sentence_transformers
import json
import streamlit as st



def get_domain_link():
    return "https://courses.analyticsvidhya.com"

def clean_text(text):
    text = text.replace("\n", "")
    text = text.replace("\t", "")
    return text.strip()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_course_details(url):
    course_texts = []
    progress_bar = st.progress(0)
    for page_no in range(1, 10):
        print("page :",page_no)
        response = requests.get(url, params={'page': page_no})
        soup = BeautifulSoup(response.content, "html.parser")

        products_list = soup.find_all('a', class_='course-card__public')
        course_links = [course_link.get("href") for course_link in products_list]

        for course_link in course_links:
            course_url = get_domain_link() + course_link
            response = requests.get(course_url)
            course_soup = BeautifulSoup(response.content, "html.parser")

            course_name = course_soup.find('h1', class_ = 'section__heading').get_text()
            course_description = course_soup.find('div', class_ = 'fr-view').get_text()

            course_curriculum_titles_raw = course_soup.find_all('h5', class_ = 'course-curriculum__chapter-title')
            course_curriculum_titles = [course_curriculum_title.get_text() for course_curriculum_title in course_curriculum_titles_raw]

            course_curriculum_lessons_raw = course_soup.find_all('span', class_ = 'course-curriculum__chapter-lesson')
            course_curriculum_lessons = [course_curriculum_lesson.get_text() for course_curriculum_lesson in course_curriculum_lessons_raw]

            course_texts.append({
                    "text": course_name,
                    "type": "course_name",
                    "link" : course_url,
                    "course_name" : course_name
                })
            course_texts.append({
                    "text": course_description,
                    "type": "course_description",
                    "link" : course_url,
                    "course_name" : course_name
                })

            for course_curriculum_title in course_curriculum_titles:
                title = clean_text(course_curriculum_title)
                course_text = {
                    "text": title,
                    "type": "title",
                    "link" : course_url,
                    "course_name" : course_name
                }
                course_texts.append(course_text)

            for course_curriculum_lesson in course_curriculum_lessons:
                lesson = clean_text(course_curriculum_lesson)
                course_text = {
                    "text": lesson,
                    "type": "lesson",
                    "link" : course_url,
                    "course_name" : course_name
                }
                course_texts.append(course_text)
    with open('content.json', 'w') as f:
        json.dump(course_texts, f, indent=4)
    return course_texts

def get_documents(course_texts):
    texts = []
    metadatas = []
    for course_text in course_texts:
        texts.append(course_text["text"])
        metadatas.append({
                        "type": course_text["type"],
                        "link" : course_text["link"],
                        "course_name" : course_text["course_name"]
                    })
    text_splitter = CharacterTextSplitter(chunk_size=1000)
    documents = text_splitter.create_documents(texts = texts, metadatas = metadatas)
    return documents

def read_json_data(file_path):
  try:
    with open(file_path, 'r') as file:
      data = json.load(file)
      return data
  except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    return None
  except json.JSONDecodeError:
    print(f"Error: Invalid JSON format in {file_path}")
    return None

def main():
    st.title("Analytics Vidhya Course Scraper")
    url = get_domain_link() + "/collections/courses"
    courses_texts = get_course_details(url)
    query = st.text_input("What do you want to learn today", value="Large language models")
    
    if st.button("Fetch Courses"):
        st.info("Fetching courses please wait...")
        courses_texts = read_json_data("course_data.json")
        documents = get_documents(courses_texts)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma.from_documents(documents, embeddings)
        docs = db.similarity_search(query)
        
        if docs:
            st.success(f"Found {len(docs)} courses!")
            st.write("Course Names and Links:")
            for i, course in enumerate(docs):
                st.write(f"{i+1}. {course.metadata['course_name']}")
                st.write(f"   -{course.metadata['link']}")
        else:
            st.warning("No courses found.")

if __name__ == "__main__":
    main()