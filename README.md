# Setup
- Create conda environment with (optional) desired python version
`conda create -n <name> python=<version>`
- Add name, version, description, etc. fields to `setup.py`. You can specify versions with `>=, ==`, e.g., `notebook==5.7.0`.
- Add any required packages to `requirements.txt`
- Change `api_name` to whatever you want the package called

# Installation
Run `pip install -e <api_name>`

# Basic Ideas of Package Structure
Basically I structure my code for the following things, in no real order:
- easy to share/reinstall. for code reuse by others but also for server usage
- separating data from code, again for server usage but also so that you can change the data and re run the code
- separating "api" from scripts the way that I do this is
- make a python package for each "api collection" (module, but not in the sense of python file).
- make a data directory and have the python packages refer to that data by using Path and a `constants.py` file that has those paths
- make a scripts dir and puts scripts in there that import the api packages and do things like any other script. The key thing is no api module uses anything defined in a script (edited) 