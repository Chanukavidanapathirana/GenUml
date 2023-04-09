# SDGP-Project
2nd Year Software Development GroupProject 2022/2023

## Project Name: GenUML
## Team Name: Team Syndicate
## Description
 This is a web application that can auto generate use case diagram from given user scenario by using machine learning and natural language processing.

## 👉 Install dependencies(To run locally)

React - 
```bash
        cd client # move to client directory
        npm install
```
Spacy-
```bash
    pip install -U pip setuptools wheel
    pip install -U spacy
    python -m spacy download en_core_web_sm
```
Gingerit-
```bash
    pip install gingerit
```
Scikit-learn
```bash
    pip install -U scikit-learn
```
Pickle
```bash
   pip install pickle5
```
Pandas
```bash
   pip install pandas
```
### Pages Created
#### Home page
#### User requirment page
#### Download page
#### Edit page 
#### Signup page
#### Signin page

## Live preview
https://genuml.onrender.com/

## Codebase tree structure

The project has a simple structure, represented as bellow:
```bash
GenUML
│    
│   
├───.github
│   └───workflows
│           Full Deployment.yml
│           superlinter.yml
│           
├───client
│   │   .gitignore
│   │   package-lock.json
│   │   package.json
│   │   README.md
│   │   
│   ├───public
│   │       favicon.ico
│   │       index.html
│   │       logo192.png
│   │       logo512.png
│   │       manifest.json
│   │       robots.txt
│   │       
│   └───src
│       │   App.css
│       │   App.js
│       │   App.test.js
│       │   index.css
│       │   index.js
│       │   logo.svg
│       │   reportWebVitals.js
│       │   setupTests.js
│       │   
│       ├───components
│       │   ├───alert
│       │   │       AlertMsg.css
│       │   │       AlertMsg.js
│       │   │       Loading.js
│       │   │       
│       │   ├───edit_items
│       │   │       Edit_items.js
│       │   │       
│       │   ├───footer
│       │   │       Footer.js
│       │   │       
│       │   └───nav
│       │       ├───accountBox
│       │       │       AccountBox.css
│       │       │       AccountBox.js
│       │       │       
│       │       └───navbar
│       │               Navbar.css
│       │               Navbar.js
│       │               
│       ├───context
│       │       DiagramDictionaryContext.js
│       │       DiagramMarkdownContext.js
│       │       
│       └───pages
│           ├───Download
│           │       Download.css
│           │       Download.js
│           │       
│           ├───Edit
│           │       Edit.css
│           │       Edit.js
│           │       
│           ├───Home
│           │       Home.css
│           │       Home.js
│           │       
│           ├───SignIn
│           │       SignIn.css
│           │       SignIn.js
│           │       
│           ├───Signup
│           │       Signup.css
│           │       Signup.js
│           │       
│           └───UserInput
│                   UserInput.css
│                   UserInput.js
│                   
├───serverside
│   │   DATASET2.csv
│   │   editDiagram.php
│   │   generateDiagram.php
│   │   mainGenerate.py
│   │   model.pkl
│   │   
│   ├───draw
│   │   │   draw_diagram.py
│   │   │   
│   │   └───__pycache__
│   │           draw_diagram.cpython-310.pyc
│   │           
│   ├───Login_Register
│   │       login.php
│   │       register.php
│   │       
│   ├───Machine_Learning_Model
│   │   │   BuildMLmodel.py
│   │   │   DATASET2.csv
│   │   │   mainMLGenerate.py
│   │   │   MLModelConnection.py
│   │   │   model.pkl
│   │   │   
│   │   └───__pycache__
│   │           mainMLGenerate.cpython-310.pyc
│   │           mainMLGenerate.cpython-311.pyc
│   │           MLConnection.cpython-310.pyc
│   │           MLModelConnection.cpython-310.pyc
│   │           MLModelConnection.cpython-311.pyc
│   │           
│   ├───NLP_Model
│   │   │   NLPmodel.py
│   │   │   
│   │   └───__pycache__
│   │           NLPmodel.cpython-310.pyc
│   │           
│   └───spelling_grammer_checker
│       │   grammar_checker.py
│       │   main.py
│       │   tester.py
│       │   
│       └───__pycache__
│               grammar_checker.cpython-310.pyc
│               main.cpython-310.pyc
│               
└───Test
        nlpTesting.py
        testing1
        testingSpellingGrammar.py
        __init__.py
        

