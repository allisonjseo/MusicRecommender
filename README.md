# Music Recommendation Application

Our model is based on this open-source repository: https://github.com/11a55an/music-recommendation-system.

Our changes include extra data preprocessing, optimizing the model for finding just the top k recommendations, and making a UI.
The original algorithm directly took song IDs as the input, but we wanted to make it more user-friendly and intuitive, so 
we implemented preprocessing steps where we took input songs and artist names as strings and using a matching algorithm to find their corresponding song IDs in our dataset.

We also created the user interface (sandbox.py) that creates a website for the users to provide the inputs and get the recommended songs from the backend model. The UI was created using Gradio, which comes with built-in themes. We experimented with different themes and layouts and got feedback from our user study to
settle on our current design.
