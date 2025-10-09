from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# File name
pdf_file = "Akhil_CV.pdf"

# Create PDF document
doc = SimpleDocTemplate(pdf_file, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)

# Styles
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Heading', fontSize=14, leading=16, spaceAfter=6, spaceBefore=6, fontName='Helvetica-Bold'))
styles.add(ParagraphStyle(name='SubHeading', fontSize=12, leading=14, spaceAfter=4, fontName='Helvetica-Bold'))
styles.add(ParagraphStyle(name='NormalText', fontSize=10, leading=12))

# Content
content = []

content.append(Paragraph("Akhil T Cheriyan", styles['Heading']))
content.append(Paragraph("Software Engineer | Kollam, Kerala", styles['NormalText']))
content.append(Paragraph("📞 9446987019 | 📧 akhiltcheriyan002@gmail.com | 🌐 github.com/AkhilATC", styles['NormalText']))
content.append(Spacer(1, 12))

content.append(Paragraph("Professional Experience", styles['SubHeading']))

# InerG Innovations
content.append(Paragraph("<b>InerG Innovations and Technologies, Kochi — Lead Engineer</b> (2023–Present) | Projects: FieldPulse, Almika, QuardINs", styles['NormalText']))
content.append(Paragraph("- Developed and maintained microservices across three major enterprise products, ensuring high scalability and reliability.", styles['NormalText']))
content.append(Paragraph("- Designed RESTful APIs using Flask and FastAPI with CRUD operations and robust error handling.", styles['NormalText']))
content.append(Paragraph("- Built Airflow DAGs and Celery-based alert/scheduler systems for real-time workflows.", styles['NormalText']))
content.append(Paragraph("- Engineered dynamic rule engine using Lark (CFG parser) for QuardINs.", styles['NormalText']))
content.append(Spacer(1, 6))

# Dexlock
content.append(Paragraph("<b>Dexlock Technologies, Kochi — Software Engineer</b> (2022–2023) | Project: Programmatic Ad Tech (RTB)", styles['NormalText']))
content.append(Paragraph("- Backend development of real-time bidding platform with high performance and scalability.", styles['NormalText']))
content.append(Paragraph("- Developed Campaign Manager module with CRUD operations, filter-based search, and REST APIs.", styles['NormalText']))
content.append(Spacer(1, 6))

# Cognicor
content.append(Paragraph("<b>Cognicor Technologies, Kochi — R&D Engineer</b> (2018–2022) | Project: CIRA", styles['NormalText']))
content.append(Paragraph("- Developed microservices for large-scale real-time applications.", styles['NormalText']))
content.append(Paragraph("- Implemented Celery-based distributed tasks and extended queues.", styles['NormalText']))
content.append(Paragraph("- Developed ontology-to-JSON knowledge base conversion module.", styles['NormalText']))
content.append(Paragraph("- Built ontology services using Apache Jena and OWLAPI for semantic reasoning.", styles['NormalText']))
content.append(Spacer(1, 6))

# Mozanta
content.append(Paragraph("<b>Mozanta Technologies, Trivandrum — Python Developer Intern</b> (2017–2018) | Project: Interviewer Chatbot", styles['NormalText']))
content.append(Paragraph("- Developed voice-interfaced chatbot with Neo4j-based knowledge ingestion.", styles['NormalText']))
content.append(Spacer(1, 12))

# Skills
content.append(Paragraph("Skills", styles['SubHeading']))
content.append(Paragraph("Python, Java, Flask, FastAPI, Spring Boot, Angular (beginner), MongoDB, Redis, RabbitMQ, Docker, ELK, OpenSearch, Airflow DAGs, Celery, Ontology (OWL, OWLAPI, Apache Jena), CFG Parsing (Lark), NLP, Machine Learning, LLM, RAG, LangChain, LlamaIndex", styles['NormalText']))
content.append(Spacer(1, 12))

# Education
content.append(Paragraph("Education", styles['SubHeading']))
content.append(Paragraph("M.Tech, Data Science and AI — CUSAT (2023–Present)", styles['NormalText']))
content.append(Paragraph("M.Sc, Computer Science (Machine Intelligence) — IIITM-K (2016–2018)", styles['NormalText']))
content.append(Paragraph("BCA, Computer Applications — University Institute of Technology, Kollam (2013–2016)", styles['NormalText']))
content.append(Spacer(1, 12))

# Projects
content.append(Paragraph("Independent Projects", styles['SubHeading']))
content.append(Paragraph("Semantic Knowledge-based Tutoring System | Petri Net Viewer | Tweet2WordCloud | Tiny Online Code Editor", styles['NormalText']))

# Build PDF
doc.build(content)
print(f"PDF '{pdf_file}' created successfully.")
