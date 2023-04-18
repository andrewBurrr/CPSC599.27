import spacy
from spacy.tokens import DocBin
from cltk.corpus.greek.tlg.parse_tlg_indices import get_female_authors
from cltk.corpus.greek.tlgu import TLGU
from sklearn.metrics import precision_score, recall_score, f1_score

nlp = spacy.load("grc_ud_proiel_trf")

# Get a list of female authors from the corpus
female_authors = get_female_authors()

# Create a new DocBin to store the annotated documents
doc_bin = DocBin()

# Iterate over the female authors and add their texts to the DocBin
for author in female_authors:
    tlg = TLGU(author)
    text = tlg.raw()
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PER":
            doc.ents = [(ent.start_char, ent.end_char, "FEM_PER")]
    doc_bin.add(doc)

# Train the GreCy model on the annotated documents
nlp.pipeline.remove("ner")
ner = nlp.create_pipe("ner")
nlp.add_pipe(ner, last=True)
ner.add_label("FEM_PER")
nlp.begin_training()
for i in range(20):
    doc_bin = DocBin().from_disk("path/to/annotated_docs")
    for doc in doc_bin.get_docs(nlp.vocab):
        gold = {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}
        nlp.update([doc], [gold], drop=0.2)

test_text = "Η Σοφία Καρβέλα είναι η πρόεδρος της Ελληνικής Εταιρείας Ενδοκρινολογίας."
doc = nlp(test_text)
true_entities = [("Σοφία Καρβέλα", "FEM_PER"), ("Ελληνική Εταιρεία Ενδοκρινολογίας", "ORG")]
pred_entities = [(ent.text, ent.label_) for ent in doc.ents]
precision = precision_score(true_entities, pred_entities, average="weighted")
recall = recall_score(true_entities, pred_entities, average="weighted")
f1 = f1_score(true_entities, pred_entities, average="weighted")
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)