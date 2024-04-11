
<div align="center">

# 🐍 Python Machine Learning Template 

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](https://choosealicense.com/licenses/mit/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?style=flat-square&logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

[![Linux](https://img.shields.io/badge/Linux-yellow?style=flat-square&logo=linux)]()
[![macOS](https://img.shields.io/badge/MacOS-inactive?style=flat-square&logo=macos)]()
[![Windows](https://img.shields.io/badge/Windows-blue?style=flat-square&logo=windows11)]()


The Python Machine Learning Template is designed to provide a comprehensive structure for end-to-end
**Machine Learning** projects in Python. Whether you're working on ***Computer Vision***, ***Natural Language Processing***, ***Reinforcement Learning***, or ***traditional Machine Learning/Data Science***, this template offers a simple and intuitive way to organize and manage your project.


<a href="https://github.com/anujonthemove/Python-Machine-Learning-Template/generate"><img src="https://raw.githubusercontent.com/anujonthemove/Python-Machine-Learning-Template/main/.assets/images/do-not-clone_fork-click-this-button-to-use-the-template.svg" alt="click this button to use the template"></a>

To get started, simply click on the above button 👆 or <b><font color="red">"Use this template"</font></b> button at the top and create your project based on this template. 

<a href="https://www.anujonthemove.com/Software-Engineering-Handbook/python-machine-learning-template/"><img src="https://raw.githubusercontent.com/anujonthemove/Python-Machine-Learning-Template/main/.assets/images/check-detailed-documentation-here.svg" alt="check detailed documentation here"></a>

</div>


The following sections provide an overview of the directory structure and instructions for setting up your project workspace.

## 🗂️ Directory Structure

```

.
├── config/                  <- 📂 Configuration files [.ini, .json, .yaml]
├── data/                    <- 📂 Images, numpy data objects, text files
├── docs/                    <- 📂 Store .md files. Used by Mkdocs for Project Documentation
├── helpers/                 <- 📂 Utility/helper files/modules for the project
├── html/                    <- 📂 Store .html files and accompanying assets. Used by pdoc3 for API Documentation
├── logs/                    <- 📂 Log files generated by the project during execution
├── models/                  <- 📂 Model files [.h5, .pkl, .pt] - pre-trained weight files, snapshots, checkpoints
├── notebooks/               <- 📂 Jupyter Notebooks
├── references/              <- 📂 Data dictionaries, manuals, and all other explanatory materials
├── scripts/                 <- 📂 Utility scripts for various project-related tasks
├── src/                     <- 📂 Source code (.py files)
├── tests/                   <- 📂 Unit tests for the project
├── workspaces/              <- 📂 Multi-user workspace that can be used in the case of a single machine
├── .env-template            <- 🔧 Template for the .env file
├── .gitattributes           <- 🔧 Standard .gitattributes file
├── .gitignore               <- 📛 Standard .gitignore file
├── .pre-commit-config.yaml  <- 🔧 Config file for Git Hooks
├── LICENSE                  <- 🪧 License file [choose your appropriate license from GitHub]
├── mkdocs.yml               <- 🗞️ Base config file required for Mkdocs
├── Pipfile		              <- 🗃️ Most commonly used python packages
├── project_setup.bat        <- 📜 Project script for Windows OS
├── project_setup.sh         <- 📜 Project script for Linux/MacOS
├── README.md                <- 📝 Project readme
├── setup.py                 <- 📦️ For installing & packaging the project
└── tox.ini                  <- 🔧 General-purpose package configuration manager

```

## 🚀 Features

* 🤓 Simple, intuitive, yet comprehensive directory structure for organizing your machine learning project.

* 😎 Setup script provided for Windows, Mac, and Linux.  Check [Setup](#%EF%B8%8F-setup) section for more details.

* 🤗 Only requires native Python and Virtual Environment package installed.

* 🤩 Uses [Pipenv](https://pipenv.pypa.io/en/latest/#). Here are some of the advantages of using it:

   * **💪 Consolidated Tooling:** By combining the functionalities of pip, virtualenv, and package management, Pipenv eliminates the need to use these tools separately.  

	* **🤌 Simplified Dependency Management:** Pipenv removes the need for the requirements.txt file and replaces it with the Pipfile, which effectively tracks dependencies. Additionally, Pipenv utilizes the Pipfile.lock to ensure consistent and reliable application builds.

	* **🤟 Cross-Platform Compatibility:** Pipenv supports a wide range of operating systems, including Linux, macOS, and Windows. 


## 🛠️ Setup

Project setup scripts have been provided separately for Linux/MacOS and Windows OS. 

Checkout the demo quick setup.

### 🎥 Demo

[![asciicast](https://asciinema.org/a/aLntFu3A4w0JWivMfXLzfVvCv.svg)](https://asciinema.org/a/aLntFu3A4w0JWivMfXLzfVvCv)


<details>
<summary> <h3> Command details </h3> </summary>

   ```
   ## For Linux/MacOS
   source project_setup.sh [OPTIONS] 

   or 

   ## For Windows OS
   project_setup.bat [OPTIONS]
   ```

   Replace [OPTIONS] with any combination of the following options:

   * `--install`: **Required argument**. If nothing is passed, a help message is displayed.
   
   * `--install-dev`: Optional argument. Pass this flag along with `--install` flag to install development packages.
   
   * `--use-proxy`: Optional argument. This flag enables installation of python packages behind proxy. Check Using .env section for proxy configuration.
   
   * `--unset-proxy`: Optional argument. This flag disables proxy.

   * `--clear-readme`: Optional argument. Clear README.md file after setting up the project.
      * 📣 ***Caution: Use this only when you are setting up the project for the first time.***

   * `--remove-cache`: Optional argument. Removes `pip` and `pipenv` cache files.
      * 💡 ***Use this to clear cache files generated during package installation***

   * `--help`: Display the help message.

</details>


<details>
<summary><h3> 🐧 Instructions for Linux/MacOS </h3></summary>

For setting up the project, `project_setup.sh` script has been provided along with some options.


### 🧑‍💻 Steps:

1. Open terminal and navigate to your project directory.

   **Case (a): Setting up in _Development_ environment** 
   
      If you are setting up the project inside **development** environment, use:

      ```
         source project_setup.sh --install --install-dev
      ```

      Incase you are working behind a proxy, use the following command instead:
         
      ```
         source project_setup.sh --install --install-dev --use-proxy
      ```
   

   **Case (b): If you are setting up the project in _production_ environment**, 
   
      If you are setting up the project inside **production** environment, you may only require base packages to be installaed, use:

      ```
         source project_setup.sh --install
      ```

      If you are working behind a proxy, use the following command:
      
      ```
         source project_setup.sh --install --use-proxy
      ```

2. If you are setting up the project first time using this template, then you should replace contents of the README.md with the name of your project:

   ```
   source project_setup.sh --clear-readme
   ``` 
   
   ***Use this command only once in the development environment. DO NOT run this once you write your own readme. Also, do not run this in production environment.***



#### 📝 Important Note 

*  For any other package installation apart from the listed packages in `Pipfile` use `pipenv` as follows:

   ```
   pipenv install package_name
   ```

   By default, `pipenv` loads all the `.env` variables, therefore you need to unset the proxy first if you are not behind proxy.

   Use the following command:

   ```
   source project_setup.sh --unset-proxy
   ```
   You should then be able to install packages using pipenv as stated above.

*  During package installation, the packages are downloaded and cached. This consumes a lot of disk, hence you should clear pip and pipenv cache from time to time. Use the following command:

   ```
   source project_setup.sh --remove-cache
   ``` 


*  ✅ To ensure a conflict-free environment setup, it is strongly recommended to always run the `project_setup.sh` script to create a virtual environment for your project.

*  ❗You should run the script **ONLY** using the `source` command to ensure that the virtual environment `.venv` is automatically activated at the end of setup in the current shell session.


</details>
<details>

<summary><h3> 🪟 Instructions for Windows OS</h3></summary>


For setting up the project, `project_setup.bat` script has been provided along with some options.


### 🧑‍💻 Steps:

1. Open Command Prompt (CMD) and navigate to your project directory.

   **Case (a): Setting up in _Development_ environment** 
   
      If you are setting up the project inside **development** environment, use:

      ```
      project_setup.bat --install --install-dev
      ```

      Incase you are working behind a proxy, use the following command instead:
         
      ```
      project_setup.bat --install --install-dev --use-proxy
      ```
   

   **Case (b): If you are setting up the project in _production_ environment**, 
   
      If you are setting up the project inside **production** environment, you may only require base packages to be installaed, use:

      ```
      project_setup.bat --install
      ```

      If you are working behind a proxy, use the following command:
      
      ```
      project_setup.bat --install --use-proxy
      ```

2. If you are setting up the project first time using this template, then you should replace contents of the README.md with the name of your project:

   ```
   project_setup.bat --clear-readme
   ``` 
   
   ***Use this command only once in the development environment. DO NOT run this once you write your own readme. Also, do not run this in production environment.***



#### 📝 Important Note 

*  For any other package installation apart from the listed packages in `Pipfile` use `pipenv` as follows:

   ```
   pipenv install package_name
   ```

   By default, `pipenv` loads all the `.env` variables, therefore you need to unset the proxy first if you are not behind proxy.

   Use the following command:

   ```
   project_setup.bat --unset-proxy
   ```
   You should then be able to install packages using pipenv as stated above.

*  During package installation, the packages are downloaded and cached. This consumes a lot of disk, hence you should clear pip and pipenv cache from time to time. Use the following command:

   ```
   project_setup.bat --remove-cache
   ``` 

*  ✅ To ensure a conflict-free environment setup, it is strongly recommended to always run the `project_setup.bat` script to create a virtual environment for your project.

*  ❗For security reasons, organizations may prevent running .bat scripts on PowerShell. You should run the script **ONLY** on Command Prompt (CMD) to ensure that everything runs without any errors.



</details>


## 📦 Packages
All the packages to be installed are included in the Pipfile. For installing additional packages **ONLY** `pipenv` should be used.

<details> 
<summary> <h3> Base Packages </h3> </summary>

```
* numpy           <- for numerical computing and scientific computing
* scipy           <- mathematical algorithms and convenience functions built on the NumPy
* pandas          <- for data manipulation and analysis
* matplotlib      <- plotting library
* seaborn         <- data visualization library for drawing informative statistical graphics.
* scikit-learn    <- machine learning library 
* jupyter         <- web-based interactive computing platform
* jupyter-server  <- backend for Jupyter notebooks. Required when running notebooks in VS Code
* ipykernel       <- interactive Python shell. Required when running notebooks in VS Code
* ipython         <- provides a powerful interactive shell and a kernel for Jupyter
```
</details>
 
<details> 
<summary> <h3> Development Packages </h3> </summary>


```
* isort                        <- sorts imports in a python file
* python-decouple              <- Reads configuration/settings from .env, system environment variables 
* flake8                       <- Code linter (format checker)
* flake8-tabs                  <- Tab (and Spaces) Style Checker for flake8
* black                        <- Code formatter
* mypy                         <- Static type checker
* pre-commit                   <- A framework for managing and maintaining multi-language pre-commit hooks.
* pdoc3                        <- Generate API documentation for Python projects
* mkdocs                       <- Generate Project documentation for Python projects
```
</details>


## 🌟 Star Us
If you find our project useful, please consider giving it a star on GitHub. 🤩

It motivates us to continue improving and adding new features. 💪

Thank you for your support ❤️

## 👥 Authors

- [@Stefanos Metzidakis](https://github.com/biodeveloper)
- [@emmanouela Xenou](https://github.com/......)
