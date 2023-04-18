import os
import glob
import re
from bs4 import BeautifulSoup
import contractions
import spacy
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from cltk.stop.greek.stops import STOPS_LIST
from cltk.corpus.greek.alphabet import expand_iota_subscript
from cltk.corpus.greek.beta_to_unicode import Replacer

def get_qualified_pairs(xml_paths):
    qualified_pairs = []
    for xml_path in xml_paths:
        match = re.search(r"(?P<name>[^_]*)_(?P<language>gk|eng).(?P<extension>xml)", xml_path)
        if match is None: continue
        if match.group("language") == "eng":
            gk_file = xml_path.replace("eng", "gk")
            if os.path.isfile(gk_file):
                qualified_pairs.append((gk_file, xml_path))
    return qualified_pairs


def perseus_tei_xml_to_text():
    r = Replacer()
    xml_dir = os.path.normpath("./data/Classics/*/*/*.xml")
    xml_paths = glob.glob(xml_dir)

    xml_pairs = get_qualified_pairs(xml_paths)
    new_dir = os.path.normpath("./data/Pairs/")
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    for xml_pair in xml_pairs:
        xml_gk, xml_eng = xml_pair
        pair_file = os.path.split(xml_gk)[-1].replace(".xml", "_en.txt")

        with open(xml_gk) as gk_file, open(xml_eng) as en_file:
            gk_soup = BeautifulSoup(gk_file, features="xml")
            en_soup = BeautifulSoup(en_file, features="xml")
        title = en_soup.title.get_text()
        author = en_soup.author.get_text()

        for note in gk_soup.find_all("note"): note.decompose()
        for note in en_soup.find_all("note"): note.decompose()
        for milestone in gk_soup.find_all("milestone"): milestone.replace_with(" ")
        for milestone in en_soup.find_all("milestone"): milestone.replace_with(" ")
        for speaker in gk_soup.find_all("speaker"): speaker.decompose()
        for speaker in en_soup.find_all("speaker"): speaker.decompose()

        gk_pars = gk_soup.findAll("p")
        en_pars = en_soup.findAll("p")

        text = f"{title}\t{author}.\n"
        for gk_text, en_text in zip(gk_pars, en_pars):
            gk_text = re.sub(r"&.*?;", "", gk_text.get_text())
            en_text = re.sub(r"&.*?;", "", en_text.get_text())
            gk_text = r.beta_code(gk_text)
            text += f"{gk_text}\t{en_text}\n"
        new_plain_text_path = os.path.join(new_dir, pair_file)
        with open(new_plain_text_path, "w+") as file_open:
            file_open.write(text)


perseus_tei_xml_to_text()
#
#
# nlp = spacy.load("grc_ud_proiel_trf")
#
# r = Replacer()
# with open("./data/Pairs/aesch.ag_gk.txt", "r") as f:
#     text = f.read()
# sample = r.beta_code(text)
# doc = nlp(sample)
# print(sum(1 for dummy in doc.sents))