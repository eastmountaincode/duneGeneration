import streamlit as st
import logging
logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

from transformers import AutoTokenizer, AutoModelForCausalLM

##tokenizer = AutoTokenizer.from_pretrained("eastmountaincode/newDuneModel")
##
##model = AutoModelForCausalLM.from_pretrained("eastmountaincode/newDuneModel")

from aitextgen import aitextgen
ai = aitextgen(model = "eastmountaincode/newDuneModel")
##ai = aitextgen(model_folder="newDuneModel",
##               to_gpu=False)
from keras.models import load_model
import autokeras as ak
model_rating = load_model('rating/content/saved_model2',
                          custom_objects=ak.CUSTOM_OBJECTS)
model_helpful = load_model('helpful/content/saved_model/my_model',
                           custom_objects=ak.CUSTOM_OBJECTS)

st.image("duneGeneratorPic-01.png")

def generateReview():
  validText = False
  while(validText == False):
    print("Generating")
    text = ai.generate_one(
                prompt="BODY:",
                max_length=512,
                top_p=0.9,
                temperature = 1.0)
    textList = text.split("\n")
    if len(textList) < 3:
      print("Don't have all three parts")
      continue
    if (not(textList[0].startswith("BODY:") and textList[1].startswith("TITLE:") and textList[2].startswith("USERNAME:"))):
      print("Bad labels")
      continue
    output = ["", "", ""]
    output[0] = textList[0][6:]
    output[1] = textList[1][7:]
    output[2] = textList[2][10:]
    if len(output[2]) > 150:
      print("Username too long")
      continue
    validText = True
  textForClassification = "TITLE: " + output[1] + "\n" + "BODY: " + output[0]
  predicted_rating = int(round(model_rating.predict([textForClassification]).item(), 1))
  predicted_helpful = int(round(model_helpful.predict([textForClassification]).item(), 1))

  generatedData = {}
  generatedData["Title"] = output[1]
  generatedData["Review"] = output[0]
  generatedData["Username"] = output[2]
  generatedData["Rating"] = predicted_rating
  generatedData["HelpfulScore"] = predicted_helpful

  st.header(generatedData["Title"])
  st.caption("Username: " + generatedData["Username"])
  st.caption("Rating: " + str(generatedData["Rating"]) + "/10")
  st.write(generatedData["Review"])
  st.caption(str(generatedData["HelpfulScore"]) + " percent of people found this review helpful")

    

def generateReview2():
  validText = False
  while(validText == False):
    print("Generating")
    text = ai.generate_one(
                prompt="BODY:",
                max_length=512,
                top_p=0.9,
                temperature = 1.0,
                num_beams = 2,
                repetition_penalty = 3.0)
    textList = text.split("\n")
    if len(textList) < 3:
      print("Don't have all three parts")
      continue
    if (not(textList[0].startswith("BODY:") and textList[1].startswith("TITLE:") and textList[2].startswith("USERNAME:"))):
      print("Bad labels")
      continue
    output = ["", "", ""]
    output[0] = textList[0][6:]
    output[1] = textList[1][7:]
    output[2] = textList[2][10:]
    if len(output[2]) > 150:
      print("Username too long")
      continue
    validText = True
    
  textForClassification = "TITLE: " + output[1] + "\n" + "BODY: " + output[0]
  predicted_rating = int(round(model_rating.predict([textForClassification]).item(), 1))
  predicted_helpful = int(round(model_helpful.predict([textForClassification]).item(), 1))

  generatedData = {}
  generatedData["Title"] = output[1]
  generatedData["Review"] = output[0]
  generatedData["Username"] = output[2]
  generatedData["Rating"] = predicted_rating
  generatedData["HelpfulScore"] = predicted_helpful

  st.header(generatedData["Title"])
  st.caption("Username: " + generatedData["Username"])
  st.caption("Rating: " + str(generatedData["Rating"]) + "/10")
  st.write(generatedData["Review"])
  st.caption(str(generatedData["HelpfulScore"]) + " percent of people found this review helpful")


generateButton = st.button("Generate review", on_click = generateReview)
generateButton2 = st.button("Generate higher quality review (may take longer)",
                            on_click = generateReview2)
  

