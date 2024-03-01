import {
    gomoku,
    skillspotter,
    nst,
    mobile,
    comp_phy,
    web,
    supermarket_sales,
    banking,
    cusine_cluster,
    hmo,
    spam,
    fig1,
    fig2,
    model,
    datae,
    ratings
  } from "../assets";

  export const navLinks = [
    {
      id: "about",
      title: "About",
    },
    {
      id: "projects",
      title: "Projects",
    },
    {
      id: "contact",
      title: "Contact",
    },
  ];
  
  const services = [
    {
      title: "Data Scientist",
      icon: web,
    },
    {
      title: "Data Engineer",
      icon: datae,
    },
    {
      title: "Data Analyst",
      icon: mobile,
    },
    {
      title: "Computational Physicist",
      icon: comp_phy,
    },
  ];
  
  const projects = [

    {
      name: "HealthCost Insight: A Data-Driven Approach to Healthcare Cost Management",
      category: ["Data Science"],
      description:
        "In this data science project, we analyzed healthcare cost data to identify key influencing factors. We used statistical modeling to understand the relationships within the data and made strategic recommendations for cost reduction based on our findings. To enhance the accessibility of our results, we developed an interactive dashboard using Shiny App in R, allowing users to visualize the data and test different models. This project showcases our ability to derive actionable insights from data and contribute to effective healthcare cost management.",
      topics: "Exploratory Data Analysis, Data Preprocessing and Cleaning, Correlation Analysis, Univariate Analysis, Bivariate Analysis, Multivariate Analysis, Linear Regression, Multiple Linear Regression, Model Evaluation (R-squared values), Data Visualization (Scatterplots, Maps), Predictive Modeling, Shiny App Development, Dashboard Creation, Data-Driven Recommendations, Model Building (Tree Bag, Support Vector Machine), Model Confusion Analysis, Data Insights Interpretation, R Programming, Statistical Analysis, Healthcare Cost Analysis",
      libraries: "R Libraries : tidyverse, imputeTS, rpart, ggplot2, usmap, dplyr, caret, shiny, corrplot, e1071, randomForest",
      link_labels: {label1: "GitHub", label2:"Report"},
      links: {
        l1: "https://github.com/kabirthakur/datascience/tree/main/HMO%20Project",
        l2: "https://github.com/kabirthakur/datascience/blob/main/HMO%20Project/report.pdf", // Add the report link here
      },
      image: hmo, // replace with your image
      source_code_link: "https://github.com/kabirthakur/datascience/tree/main/HMO%20Project",
    },
    {
      "name": "SkillSpotter: Mining Skills from Job Descriptions using NER",
      "category": ["Data Science","NLP","Deep Learning"] ,
      "description":"This paper presents ”SkillSpotter”, a novel approach to extracting and categorizing skills from job descriptions using advanced text mining techniques. Utilizing Named Entity Recognition (NER), the study leverages a fine-tuned BERT model that accurately identifies both general and specific technological and soft skills from a vast dataset of web-scraped job descriptions. The project addresses challenges in data sampling, annotation, and model optimization while considering the ethical implications of data usage and technology in human resource contexts. The findings offer valuable insights into job market trends and skill demands, highlighting the potential of NER in job recommendation systems.",
      "topics": "Named Entity Recognition, BERT, Text Mining, Machine Learning, Data Cleaning, Data Sampling, Model Optimization, Job Market Trends Analysis, Skill Demand Analysis, Recommendation Systems, NLU, NLI, Data Transformation, Model Training and Inference, Skill Taxonomy Development, Data Annotation, Pattern Matching",
      "libraries": "Python Libraries: PyTorch, transformers, spacy, NLTK, Pandas, datasets, seqeval, BeautifulSoup 4",
      "link_labels": { "label1": "Project Repository", "label2": "Report" },
      "links": {
        "l1": "https://github.com/kabirthakur/datascience/tree/main/NER%20on%20Job%20Descriptions",
        "l2": "https://github.com/kabirthakur/datascience/blob/main/NER%20on%20Job%20Descriptions/SkillSpotter_Report.pdf"
      },
      "image": skillspotter, // replace with generated image
      "source_code_link": "https://github.com/kabirthakur/datascience/tree/main/NER%20on%20Job%20Descriptions"
    },
    {
      "name": "Analysis of Cancelled Netflix Shows",
      "category": ["Data Analysis", "Data Engineering"],
      "description":
        "This project examines the reasons behind the cancellation of Netflix shows, leveraging MongoDB for efficient data storage and PySpark for scalable data processing. The goal is to develop a scalable framework to analyse hypothesis and uncover trends affecting show longevity. The project employs statistical analysis and visualization techniques to explore the correlation between show characteristics and their cancellations, aiming to derive actionable insights for content strategy optimization in digital streaming platforms.",
      "topics": "Data Collection, Data Cleaning, Exploratory Data Analysis, Statistical Analysis, Trend Analysis, Predictive Modeling, Visualization, Ratings Analysis, Big Data Processing, NoSQL Database Management",
      "libraries": "Python Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, PySpark; Database: MongoDB",
      "link_labels": {"label1": "GitHub Repository", "label2":"Project Report"},
      "links": {
        "l1": "hhttps://github.com/kabirthakur/datascience/tree/main/Netflix%20Cancelled%20Shows%20Analysis"
      },
      "image": ratings, // Replace with your image
      "source_code_link": "https://github.com/YourGitHub/NetflixCancellationAnalysis"
    },
    {
      "name": "Experiments In Neural Style Transfer",
      "category": ["Deep Learning"] ,
      "description":"This paper explores the intersection of artificial intelligence and art by implementing and experimenting with Neural Style Transfer (NST) algorithms. The objective is to blend the content of one image with the artistic style of another, creating unique art pieces. It involves an in-depth investigation of various optimizers like Adam and L-BFGS and pre-trained neural network models, such as VGG19 and AlexNet, to evaluate their impact on NST outcomes. The aim of this paper is to explore the underlying mathematics of NST by manipulating the loss function and experimenting with hyperparameters to create a spectrum of stylized images. Through this project, the authors seek to understand how different neural network architectures and NST parameters contribute to the art generation process, with the ultimate goal of achieving a nuanced understanding of NST in artistic creation.",
      "topics": "Neural Style Transfer (NST), PyTorch, Adam Optimizer, L-BFGS Optimizer, VGG19, AlexNet",
      "libraries": "Python Libraries: PyTorch, torchvision, ",
      "link_labels": { "label1": "Project Repository", "label2": "Report" },
      "links": {
        "l1": "https://github.com/kabirthakur/datascience/tree/main/Neural%20Style%20Transfer%20Experiments",
        "l2": "https://github.com/kabirthakur/datascience/blob/main/Neural%20Style%20Transfer%20Experiments/Neural_Artistry_A_Journey_through_Style_Transfer.pdf"
      },
      "image": nst, // replace with generated image
      "source_code_link": "https://github.com/kabirthakur/datascience/tree/main/Neural%20Style%20Transfer%20Experiments"
    },
    {
      "name": "Gomoku Using a Value Neural Network",
      "category": ["Deep Learning"] ,
      "description":"This study introduces GomokuAI, an artificial intelligence designed to play the board game Gomoku. The AI combines hard-coded strategies with a neural network-based heuristic approach. Developed using a dataset from 2000 simulated games of minimax algorithm with alpha beta pruning playing against itself, the AI prioritizes winning moves, blocks opponent wins, and strategically responds to various game situations. The neural network, implemented using PyTorch, enhances the AI's decision-making capabilities. Initial results indicate that GomokuAI significantly outperforms minimax, achieving a win or draw rate of 60-70%. The paper suggests future improvements could include further model optimization and integration with advanced algorithms like Monte Carlo Tree Search. This research demonstrates the potential of combining traditional game strategies with machine learning techniques in game AI development.",
      "topics": "AI, Neural Networks, CNN, Monte Carlo Tree Search, Value Network, AI, CNN, GOMOKU, Heuristic Analysis",
      "libraries": "Python Libraries: PyTorch ",
      "link_labels": { "label1": "Project Repository", "label2": "Report" },
      "links": {
        "l1": "https://github.com/kabirthakur/datascience/tree/main/Gomoku%20using%20ConvNets",
        "l2": "https://github.com/kabirthakur/datascience/blob/main/Gomoku%20using%20ConvNets/Report.pdf"
      },
      "image": gomoku, // replace with generated image
      "source_code_link": "https://github.com/kabirthakur/datascience/tree/main/Gomoku%20using%20ConvNets"
    },
    {
      name: "Dynamic Human AI Collaboration",
      category: ["Machine Learning","Data Science"],
      description: "In this project, I worked with the Decision Science team at Jaypee Morgan London to proposed a Bayesian framework for human-in-the-loop pipelines, where the final decision is a combination of algorithm and expert opinions. We argued that updating expert opinion priors with information sharing between experts is key to achieving superior performance. We utilized the SA Heart dataset to predict whether a patient has a coronary heart disease. We proposed a Bayesian framework for human-in-the-loop systems and leveraged collaborative intelligence. We also showed that deferral systems are a special case where the combining expert opinions follows a categorical distribution.",
      topics: "Human-in-the-loop, Machine Learning, Bayesian Framework, Collaborative Intelligence, Deferral Systems, Expert Opinions, Information Sharing",
      libraries: "Python : PyMC3, sklearn, theano, statsmodels, SciPy, NumPy, Pandas, Matplotlib, Seaborn, arviz",
      link_labels: {label1: "ICLR Publication"},
      links: {
        l1: "https://openreview.net/pdf?id=Muwb2KohnX", // Add the report link here
      },
      image: model, // replace with the link to your project image
      source_code_link: "https://openreview.net/pdf?id=Muwb2KohnX" // replace with your GitHub repository link
    },
    {
      name: "CusineCluster: Restaurant Recommendation System",
      category: ["Machine Learning","Data Science"],
      description:
        "This project presents a hybrid recommendation system for restaurants, leveraging K-means clustering and Alternating Least Squares (ALS) based Collaborative Filtering. The system was trained on a dataset of over 8 million individual reviews, segmenting restaurants into six distinct clusters. The resulting model demonstrated high accuracy in predicting user ratings, offering potential for personalized and effective restaurant recommendations.",
      topics: "PySpark, Data Preprocessing, Feature Engineering, K-means Clustering, Collaborative Filtering, ALS, Hybrid Recommendation Engine",
      libraries: "Python Libraries : Pyspark, Scikit-Learn, Pandas, Matplotlib, Seaborn",
      link_labels: {label1: "GitHub", label2:"Report"},
      links: {
        l1: "https://github.com/kabirthakur/datascience/tree/main/Yelp%20recommendation%20engine",
        l2: "https://github.com/kabirthakur/datascience/blob/main/Yelp%20recommendation%20engine/Report.pdf", // Add the report link here
      },
      image: cusine_cluster, // replace with your image
      source_code_link: "https://github.com/kabirthakur/datascience/tree/main/Yelp%20recommendation%20engine",
    },
    {
      name: "StoreSage: Analyzing Supermarket Data for Enhanced Retail Strategies in Myanmar",
      category: ["Data Analysis"],
      description: "This project analyzes sales data from three supermarkets in Myanmar, providing insights on store performance, customer experience, and product categories. It identifies opportunities to improve satisfaction, diversify offerings, and optimize strategies for profitability. Valuable guidance for enhancing retail performance is derived from gender and time trends.",
      topics: "Data Cleaning and Preprocessing, Exploratory Data Analysis (EDA), Correlation Analysis, Data Visualization, Statistical Analysis, Jupyter Notebook, GitHub",
      libraries: "Python Libraries : Pandas, Matplotlib, Seaborn",
      link_labels: {label1: "GitHub", label2:"Report"},
      links: {
        l1: "https://github.com/kabirthakur/datascience/tree/main/supermarket%20sales%20",
        l2: "https://github.com/kabirthakur/datascience/tree/main/supermarket%20sales%20/report.pdf", // Add the report link here
      },
      image: supermarket_sales, // replace with your image
      source_code_link: "https://github.com/kabirthakur/datascience/tree/main/supermarket%20sales%20",
    },
    {
      name: "Banking Behaviors : An Association Rule Expedition",
      category: ["Machine Learning","Data Science"],
      description:
        "This project analyzes sales data from three supermarkets in Myanmar, providing insights on store performance, customer experience, and product categories. It identifies opportunities to improve satisfaction, diversify offerings, and optimize strategies for profitability. Valuable guidance for enhancing retail performance is derived from gender and time trends.",
      topics: " One Hot Encoding, Data Cleaning, Data Visualization, Discretization, Data Segmentation, Association Rule Mining",
      libraries: "Python Libraries : Pandas, mlxtend, Numpy, Scikit-learn, networkx",
      link_labels: {label1: "GitHub", label2:"Report"},
      links: {
        l1: "https://github.com/kabirthakur/datascience/tree/main/association%20rules%20mining",
        l2: "https://github.com/kabirthakur/datascience/blob/main/association%20rules%20mining/report.pdf", // Add the report link here
      },
      image: banking, // replace with your image
      source_code_link: "https://github.com/kabirthakur/datascience/tree/main/association%20rules%20mining",
    },
    {
      name: "MailSift: NLP-Driven Spam Detection",
      category: ["NLP"],
      description:
        "In this NLP project, we analyzed the Enron public email corpus to build classifiers for identifying spam emails. We utilized the Natural Language Tool Kit (NLTK) library in Python to tokenize emails and extract features such as unigrams, bigrams, trigrams, and part-of-speech tags. We also used a subjectivity lexicon to count the number of positive and negative words in the emails. These features were then used to train a Naive Bayes Classifier and a Support Vector Machine (SVM). The performance of the classifiers was evaluated using measures such as accuracy, precision, recall, and F1 score. This project demonstrates our ability to apply NLP techniques and machine learning to develop effective spam filters.",
      topics: "Natural Language Processing, Text Tokenization, Feature Extraction, Unigram Features, Bigram Features, Trigram Features, Part-of-Speech Tagging, Subjectivity Lexicon, Naive Bayes Classifier, Support Vector Machine, Model Evaluation (Accuracy, Precision, Recall, F1 Score), Spam Email Classification",
      libraries: "Python Libraries: NLTK, SciKit Learn",
      link_labels: {label1: "GitHub", label2:"Report"},
      links: {
        l1: "https://github.com/kabirthakur/datascience/tree/main/Spam%20Detector%20Project",
        l2: "https://github.com/kabirthakur/datascience/blob/main/Spam%20Detector%20Project/Report.pdf", // Add the report link here
      },
      image: spam, // replace with your image
      source_code_link: "https://github.com/kabirthakur/datascience/tree/main/Spam%20Detector%20Project", // replace with your GitHub repository link
    },
    {
      name: "Quantum Wave Function Simulation",
      category: ["Computational Physics"],
      description: "This project simulates the time evolution of a quantum wave function as it interacts with various potential barriers. The wave function is initially a Gaussian wave packet, and the potential can be a single barrier, multiple barriers, or a step potential. The simulation is visualized using matplotlib's animation function.",
      topics: "Quantum Mechanics, Wave Function, Schrodinger Equation, Computational Physics",
      libraries: "Python Libraries : numpy, matplotlib",
      link_labels: {label1: "GitHub"},
      links: {
        l1: "https://github.com/kabirthakur/ComputationalPhysics/tree/main/Wave%20Function%20Simulation" // Add the report link here
      },
      image: fig1, // replace with the link to your project image
      source_code_link: "https://github.com/kabirthakur/ComputationalPhysics/tree/main/Wave%20Function%20Simulation" // replace with your GitHub repository link
    },
    {
      name: "Phase Transformations in Metallic Alloys",
      category: ["Computational Physics"],
      description: "This project uses the Finite Difference Method and the Spectral Method, along with the Euler Explicit method, to study phase transformations in metallic alloys by solving the Cahn-Hilliard equation. The theoretical basis for thermodynamic models is discussed, and a modified version of Fick's second law of diffusion is used to account for energy related to interfaces. The methods are implemented and optimized in MATLAB, and used to simulate microstructures due to phase separation caused by various initial composition profiles.",
      topics: "Phase Transformations, Metallic Alloys, Cahn-Hilliard Equation, Finite Difference Method, Spectral Method, Euler Explicit Method, Thermodynamic Models, Fick's Second Law of Diffusion.",
      libraries: "MATLAB",
      link_labels: {label1: "GitHub", label2:"Thesis"},
      links: {
        l1: "https://github.com/kabirthakur/ComputationalPhysics/tree/main/Metallic%20glass%20simulation",
        l2: "https://github.com/kabirthakur/ComputationalPhysics/blob/main/Metallic%20glass%20simulation/thesis.pdf" // Add the report link here
      },
      image: fig2, // replace with the link to your project image
      source_code_link: "https://github.com/username/repository" // replace with your GitHub repository link
    }
    
    
  ];
  export { services, projects }; //technologies, experiences, testimonials