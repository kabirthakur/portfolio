import {
    mobile,
    comp_phy,
    creator,
    web,
    javascript,
    typescript,
    html,
    css,
    reactjs,
    redux,
    tailwind,
    nodejs,
    mongodb,
    git,
    figma,
    docker,
    meta,
    starbucks,
    tesla,
    shopify,
    carrent,
    jobit,
    supermarket_sales,
    banking,
    cusine_cluster,
    hmo,
    spam,
    tripguide,
    threejs,
    fig1,
    fig2,
    model,
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
      title: "Data Analyst",
      icon: mobile,
    },
    {
      title: "Computational Physicist",
      icon: comp_phy,
    },
  ];
  
  // const technologies = [
  //   {
  //     name: "HTML 5",
  //     icon: html,
  //   },
  //   {
  //     name: "CSS 3",
  //     icon: css,
  //   },
  //   {
  //     name: "JavaScript",
  //     icon: javascript,
  //   },
  //   {
  //     name: "TypeScript",
  //     icon: typescript,
  //   },
  //   {
  //     name: "React JS",
  //     icon: reactjs,
  //   },
  //   {
  //     name: "Redux Toolkit",
  //     icon: redux,
  //   },
  //   {
  //     name: "Tailwind CSS",
  //     icon: tailwind,
  //   },
  //   {
  //     name: "Node JS",
  //     icon: nodejs,
  //   },
  //   {
  //     name: "MongoDB",
  //     icon: mongodb,
  //   },
  //   {
  //     name: "Three JS",
  //     icon: threejs,
  //   },
  //   {
  //     name: "git",
  //     icon: git,
  //   },
  //   {
  //     name: "figma",
  //     icon: figma,
  //   },
  //   {
  //     name: "docker",
  //     icon: docker,
  //   },
  // ];
  
  // const experiences = [
  //   {
  //     title: "React.js Developer",
  //     company_name: "Starbucks",
  //     icon: starbucks,
  //     iconBg: "#383E56",
  //     date: "March 2020 - April 2021",
  //     points: [
  //       "Developing and maintaining web applications using React.js and other related technologies.",
  //       "Collaborating with cross-functional teams including designers, product managers, and other developers to create high-quality products.",
  //       "Implementing responsive design and ensuring cross-browser compatibility.",
  //       "Participating in code reviews and providing constructive feedback to other developers.",
  //     ],
  //   },
  //   {
  //     title: "React Native Developer",
  //     company_name: "Tesla",
  //     icon: tesla,
  //     iconBg: "#E6DEDD",
  //     date: "Jan 2021 - Feb 2022",
  //     points: [
  //       "Developing and maintaining web applications using React.js and other related technologies.",
  //       "Collaborating with cross-functional teams including designers, product managers, and other developers to create high-quality products.",
  //       "Implementing responsive design and ensuring cross-browser compatibility.",
  //       "Participating in code reviews and providing constructive feedback to other developers.",
  //     ],
  //   },
  //   {
  //     title: "Web Developer",
  //     company_name: "Shopify",
  //     icon: shopify,
  //     iconBg: "#383E56",
  //     date: "Jan 2022 - Jan 2023",
  //     points: [
  //       "Developing and maintaining web applications using React.js and other related technologies.",
  //       "Collaborating with cross-functional teams including designers, product managers, and other developers to create high-quality products.",
  //       "Implementing responsive design and ensuring cross-browser compatibility.",
  //       "Participating in code reviews and providing constructive feedback to other developers.",
  //     ],
  //   },
  //   {
  //     title: "Full stack Developer",
  //     company_name: "Meta",
  //     icon: meta,
  //     iconBg: "#E6DEDD",
  //     date: "Jan 2023 - Present",
  //     points: [
  //       "Developing and maintaining web applications using React.js and other related technologies.",
  //       "Collaborating with cross-functional teams including designers, product managers, and other developers to create high-quality products.",
  //       "Implementing responsive design and ensuring cross-browser compatibility.",
  //       "Participating in code reviews and providing constructive feedback to other developers.",
  //     ],
  //   },
  // ];
  
  // const testimonials = [
  //   {
  //     testimonial:
  //       "I thought it was impossible to make a website as beautiful as our product, but Rick proved me wrong.",
  //     name: "Sara Lee",
  //     designation: "CFO",
  //     company: "Acme Co",
  //     image: "https://randomuser.me/api/portraits/women/4.jpg",
  //   },
  //   {
  //     testimonial:
  //       "I've never met a web developer who truly cares about their clients' success like Rick does.",
  //     name: "Chris Brown",
  //     designation: "COO",
  //     company: "DEF Corp",
  //     image: "https://randomuser.me/api/portraits/men/5.jpg",
  //   },
  //   {
  //     testimonial:
  //       "After Rick optimized our website, our traffic increased by 50%. We can't thank them enough!",
  //     name: "Lisa Wang",
  //     designation: "CTO",
  //     company: "456 Enterprises",
  //     image: "https://randomuser.me/api/portraits/women/6.jpg",
  //   },
  // ];
  
  const projects = [
    {
      name: "StoreSage: Analyzing Supermarket Data for Enhanced Retail Strategies in Myanmar",
      category: "Data Analysis",
      description: "This project analyzes sales data from three supermarkets in Myanmar, providing insights on store performance, customer experience, and product categories. It identifies opportunities to improve satisfaction, diversify offerings, and optimize strategies for profitability. Valuable guidance for enhancing retail performance is derived from gender and time trends.",
      topics: "Data Cleaning and Preprocessing, Exploratory Data Analysis (EDA), Correlation Analysis, Data Visualization, Statistical Analysis, Jupyter Notebook, GitHub",
      libraries: "Python Libraries : Pandas, Matplotlib, Seaborn",
      image: supermarket_sales, // replace with your image
      source_code_link: "https://github.com/",
    },
    {
      name: "CusineCluster: Restaurant Recommendation System",
      category: "Machine Learning",
      description:
        "This project presents a hybrid recommendation system for restaurants, leveraging K-means clustering and Alternating Least Squares (ALS) based Collaborative Filtering. The system was trained on a dataset of over 8 million individual reviews, segmenting restaurants into six distinct clusters. The resulting model demonstrated high accuracy in predicting user ratings, offering potential for personalized and effective restaurant recommendations.",
      topics: "PySpark, Data Preprocessing, Feature Engineering, K-means Clustering, Collaborative Filtering, ALS, Hybrid Recommendation Engine",
      libraries: "Python Libraries : Pyspark, Scikit-Learn, Pandas, Matplotlib, Seaborn",
      image: cusine_cluster, // replace with your image
      source_code_link: "https://github.com/",
    },
    {
      name: "Banking Behaviors : An Association Rule Expedition",
      category: "Machine Learning",
      description:
        "This project analyzes sales data from three supermarkets in Myanmar, providing insights on store performance, customer experience, and product categories. It identifies opportunities to improve satisfaction, diversify offerings, and optimize strategies for profitability. Valuable guidance for enhancing retail performance is derived from gender and time trends.",
      topics: " One Hot Encoding, Data Cleaning, Data Visualization, Discretization, Data Segmentation, Association Rule Mining",
      libraries: "Python Libraries : Pandas, mlxtend, Numpy, Scikit-learn, networkx",
      image: banking, // replace with your image
      source_code_link: "https://github.com/",
    },
    {
      name: "HealthCost Insight: A Data-Driven Approach to Healthcare Cost Management",
      category: "Data Science",
      description:
        "In this data science project, we analyzed healthcare cost data to identify key influencing factors. We used statistical modeling to understand the relationships within the data and made strategic recommendations for cost reduction based on our findings. To enhance the accessibility of our results, we developed an interactive dashboard using Shiny App in R, allowing users to visualize the data and test different models. This project showcases our ability to derive actionable insights from data and contribute to effective healthcare cost management.",
      topics: "Exploratory Data Analysis, Data Preprocessing and Cleaning, Correlation Analysis, Univariate Analysis, Bivariate Analysis, Multivariate Analysis, Linear Regression, Multiple Linear Regression, Model Evaluation (R-squared values), Data Visualization (Scatterplots, Maps), Predictive Modeling, Shiny App Development, Dashboard Creation, Data-Driven Recommendations, Model Building (Tree Bag, Support Vector Machine), Model Confusion Analysis, Data Insights Interpretation, R Programming, Statistical Analysis, Healthcare Cost Analysis",
      libraries: "R Libraries : tidyverse, imputeTS, rpart, ggplot2, usmap, dplyr, caret, shiny, corrplot, e1071, randomForest",
      image: hmo, // replace with your image
      source_code_link: "https://github.com/",
    },
    {
      name: "MailSift: NLP-Driven Spam Detection",
      category: "Natural Language Processing",
      description:
        "In this NLP project, we analyzed the Enron public email corpus to build classifiers for identifying spam emails. We utilized the Natural Language Tool Kit (NLTK) library in Python to tokenize emails and extract features such as unigrams, bigrams, trigrams, and part-of-speech tags. We also used a subjectivity lexicon to count the number of positive and negative words in the emails. These features were then used to train a Naive Bayes Classifier and a Support Vector Machine (SVM). The performance of the classifiers was evaluated using measures such as accuracy, precision, recall, and F1 score. This project demonstrates our ability to apply NLP techniques and machine learning to develop effective spam filters.",
      topics: "Natural Language Processing, Text Tokenization, Feature Extraction, Unigram Features, Bigram Features, Trigram Features, Part-of-Speech Tagging, Subjectivity Lexicon, Naive Bayes Classifier, Support Vector Machine, Model Evaluation (Accuracy, Precision, Recall, F1 Score), Spam Email Classification",
      libraries: "Python Libraries: NLTK, SciKit Learn",
      image: spam, // replace with your image
      source_code_link: "https://github.com/", // replace with your GitHub repository link
    },
    {
      name: "Quantum Wave Function Simulation",
      category: "Computational Physics",
      description: "This project simulates the time evolution of a quantum wave function as it interacts with various potential barriers. The wave function is initially a Gaussian wave packet, and the potential can be a single barrier, multiple barriers, or a step potential. The simulation is visualized using matplotlib's animation function.",
      topics: "Quantum Mechanics, Wave Function, Schrodinger Equation, Computational Physics",
      libraries: "Python Libraries : numpy, matplotlib",
      image: fig1, // replace with the link to your project image
      source_code_link: "https://github.com/username/repository" // replace with your GitHub repository link
    },
    {
      name: "Phase Transformations in Metallic Alloys",
      category: "Computational Physics",
      description: "This project uses the Finite Difference Method and the Spectral Method, along with the Euler Explicit method, to study phase transformations in metallic alloys by solving the Cahn-Hilliard equation. The theoretical basis for thermodynamic models is discussed, and a modified version of Fick's second law of diffusion is used to account for energy related to interfaces. The methods are implemented and optimized in MATLAB, and used to simulate microstructures due to phase separation caused by various initial composition profiles.",
      topics: "Phase Transformations, Metallic Alloys, Cahn-Hilliard Equation, Finite Difference Method, Spectral Method, Euler Explicit Method, Thermodynamic Models, Fick's Second Law of Diffusion.",
      libraries: "MATLAB",
      image: fig2, // replace with the link to your project image
      source_code_link: "https://github.com/username/repository" // replace with your GitHub repository link
    },
    {
      name: "Dynamic Human AI Collaboration",
      category: "Machine Learning",
      description: "In this project, my co-authors and I proposed a Bayesian framework for human-in-the-loop pipelines, where the final decision is a combination of algorithm and expert opinions. We argued that updating expert opinion priors with information sharing between experts is key to achieving superior performance. We utilized the SA Heart dataset to predict whether a patient has a coronary heart disease. We proposed a Bayesian framework for human-in-the-loop systems and leveraged collaborative intelligence. We also showed that deferral systems are a special case where the combining expert opinions follows a categorical distribution.",
      topics: "Human-in-the-loop, Machine Learning, Bayesian Framework, Collaborative Intelligence, Deferral Systems, Expert Opinions, Information Sharing",
      libraries: "Python : PyMC3, sklearn, theano, statsmodels, SciPy, NumPy, Pandas, Matplotlib, Seaborn, arviz",
      image: model, // replace with the link to your project image
      source_code_link: "https://github.com/username/repository" // replace with your GitHub repository link
    }
    
    
  ];
  export { services, projects }; //technologies, experiences, testimonials