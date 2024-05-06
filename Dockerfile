# Currently tested & workong for Python 3.11
FROM --platform=linux/amd64 python:3.9-slim

# Copy the current directory contents into the container at /app
COPY ./ /repo

# Copy and install the requirements
#COPY ./requirements.txt /requirements.txt

# Update default packages
RUN apt-get -qq update

RUN apt-get install -y -q \
    build-essential \
    curl



# Pip install the dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /repo/requirements.txt

# Set the working directory to /app
WORKDIR /repo

# Expose port 8501
EXPOSE 40000

# Run the app
#CMD streamlit run /repo/app/01_Ask.py
