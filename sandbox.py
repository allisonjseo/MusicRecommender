import gradio as gr
from predict import pred, find_song_ids

custom_css = """
body {
    background-color: #f0f8ff !important;  /* Alice Blue background for a fresh feel */
    font-family: 'Poppins', sans-serif !important;  /* A modern, clean font */
    color: #333 !important;
}

button.gr-button {
    background-color: #ff5722 !important;  /* Vivid orange color for buttons */
    color: #ffffff !important;
    border-radius: 25px !important;
    font-size: 16px !important;
    padding: 12px 24px !important;
    transition: background-color 0.3s, box-shadow 0.3s !important;
    border: none !important;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1) !important;
}

button.gr-button:hover {
    background-color: #e64a19 !important;  /* Darker orange when hovering */
    box-shadow: 0px 6px 10px rgba(0, 0, 0, 0.15) !important;
}

.gr-textbox {
    border-radius: 15px !important;
    padding: 10px !important;
    font-size: 16px !important;
    border: 2px solid #4caf50 !important;  /* Green borders to match the vibrant color scheme */
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    margin-bottom: 15px !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}

.gr-textbox:focus {
    outline: none !important;
    border-color: #ff9800 !important;  /* Orange color on focus */
    box-shadow: 0 0 8px rgba(255, 152, 0, 0.5) !important;
}

.gr-markdown {
    font-size: 20px !important;
    padding: 30px !important;
    text-align: center !important;
    background-color: #ffffff !important;
    color: #3f51b5 !important;  /* Indigo color for the text */
    border-radius: 15px !important;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.15) !important;
    margin-bottom: 25px !important;
}

.gr-container {
    padding: 40px !important;
    max-width: 800px !important;
    margin: 0 auto !important;
}

.gr-blocks > div {
    padding: 20px !important;
}

.gr-button {
    font-weight: bold !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        gr.Markdown('''# 🎵 Music Recommender 🎶
                       ### By: AJ Seo and Jeffrey Li
                       ---''')
    with gr.Row():
        gr.Markdown(
            """
            ## How It Works
            Our recommendation system makes decisions by intelligently weighting several features of the input data. These features include:
            
            - **Genre Representation**: Captures the unique and important characteristics of a song's genre.
            - **Audio Features**: Quantifies the acoustic properties of tracks, such as energy, danceability, and valence.
            - **Release Year**: Accounts for temporal trends and user preferences for music from specific time periods.
            - **Similarity Score**: Uses cosine similarity score to compare and identify songs most similar to the user's preferences.
            
            ---
            """
        )

    with gr.Row():
        gr.Markdown('''
                       ### Input **song titles**, their **artists**, and the **number of recommendations**.
                       Get personalized song recommendations!''')

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Song 1")
            song1_title = gr.Textbox(label="Title", placeholder="enter song title")
            song1_artist = gr.Textbox(label="Artist", placeholder="enter artist name")

        with gr.Column():
            gr.Markdown("### Song 2")
            song2_title = gr.Textbox(label="Title", placeholder="enter song title")
            song2_artist = gr.Textbox(label="Artist", placeholder="enter artist name")

        with gr.Column():
            gr.Markdown("### Song 3")
            song3_title = gr.Textbox(label="Title", placeholder="enter song title")
            song3_artist = gr.Textbox(label="Artist", placeholder="enter artist name")

    with gr.Row():
        num_recs = gr.Textbox(label="Number of Recommendations", placeholder="enter number of recs (e.g., 5)")

    with gr.Row():
        submit = gr.Button("🎶 Get Recommendations")

    output = gr.Textbox(label="Recommended Song(s)", placeholder="Your recommendations will appear here.")

    submit.click(fn=find_song_ids, 
                 inputs=[song1_title, song1_artist, song2_title, song2_artist, song3_title, song3_artist, num_recs], 
                 outputs=output)
    
    with gr.Row():
        gr.Markdown(
            """
            ---
            ## Further Inquiries
            We would be happy to share more details about how our model works! Feel free to reach out to us:
            
            - **AJ Seo**: [ajseo@andrew.cmu.edu](mailto:ajseo@andrew.cmu.edu)  
            - **Jeffrey Li**: [jhl4@andrew.cmu.edu](mailto:jhl4@andrew.cmu.edu)
            ---
            """
        )




demo.launch(share=True)


# import gradio as gr
# from predict import pred, find_song_ids

# custom_css = """
# body {
#     background-color: #f0f8ff !important;  /* Alice Blue background for a fresh feel */
#     font-family: 'Poppins', sans-serif !important;  /* A modern, clean font */
#     color: #333 !important;
# }

# button.gr-button {
#     background-color: #ff5722 !important;  /* Vivid orange color for buttons */
#     color: #ffffff !important;
#     border-radius: 25px !important;
#     font-size: 16px !important;
#     padding: 12px 24px !important;
#     transition: background-color 0.3s, box-shadow 0.3s !important;
#     border: none !important;
#     box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1) !important;
# }

# button.gr-button:hover {
#     background-color: #e64a19 !important;  /* Darker orange when hovering */
#     box-shadow: 0px 6px 10px rgba(0, 0, 0, 0.15) !important;
# }

# .gr-textbox {
#     border-radius: 15px !important;
#     padding: 10px !important;
#     font-size: 16px !important;
#     border: 2px solid #4caf50 !important;  /* Green borders to match the vibrant color scheme */
#     box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1) !important;
#     margin-bottom: 15px !important;
#     transition: border-color 0.3s, box-shadow 0.3s !important;
# }

# .gr-textbox:focus {
#     outline: none !important;
#     border-color: #ff9800 !important;  /* Orange color on focus */
#     box-shadow: 0 0 8px rgba(255, 152, 0, 0.5) !important;
# }

# .gr-markdown {
#     font-size: 20px !important;
#     padding: 30px !important;
#     text-align: center !important;
#     background-color: #ffffff !important;
#     color: #3f51b5 !important;  /* Indigo color for the text */
#     border-radius: 15px !important;
#     box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.15) !important;
#     margin-bottom: 25px !important;
# }

# .gr-container {
#     padding: 40px !important;
#     max-width: 800px !important;
#     margin: 0 auto !important;
# }

# .gr-blocks > div {
#     padding: 20px !important;
# }

# .gr-button {
#     font-weight: bold !important;
# }
# """


# with gr.Blocks(theme=gr.themes.Soft()) as demo:
#     with gr.Row():
#         instructions = gr.Markdown('''# 🎵 Music Recommender 🎶
#                                     ### By: AJ Seo and Jeffrey Li
#                                     ---
#                                     ### To use this tool, input **3 different song IDs** and specify the **number of recommendations** you want.
#                                     The model will provide personalized recommendations based on your input songs.
#                                     ''')
#     with gr.Row():
#         text1 = gr.Textbox(label="Song ID #1", placeholder="enter song ID 1 here")
#         text2 = gr.Textbox(label="Song ID #2", placeholder="enter song ID 2 here")
#         text3 = gr.Textbox(label="Song ID #3", placeholder="enter song ID 3 here")
#         text4 = gr.Textbox(label="Number of Recs", placeholder="enter number of recs (e.g., 5)")
        
#     # with gr.Row():
#     #     text1 = gr.Textbox(label="Track #1 Name", placeholder="enter song 1 here")
#     #     text5 = gr.Textbox(label="Track #1 Artist", placeholder="enter song 1's artist here")
#     # with gr.Row():
#     #     text2 = gr.Textbox(label="Song 2", placeholder="enter song 2 here")
#     #     text6 = gr.Textbox(label="Song 2", placeholder="enter song 2 here")
#     # with gr.Row():
#     #     text3 = gr.Textbox(label="Song 3", placeholder="enter song 3 here")
#     #     text7 = gr.Textbox(label="Song 3", placeholder="enter song 3 here")
#     # with gr.Row():
#     #     text4 = gr.Textbox(label="Number of Recs", placeholder="enter number of recs (e.g., 5)")
#     with gr.Row():
#         submit = gr.Button("🎶 Get Recommendations")

#     output = gr.Textbox(label="Recommended Song(s)", placeholder="the recommended songs will appear here...")

#     submit.click(fn=find_song_ids, inputs=[text1, text2, text3, text4], outputs=output)

# demo.launch(share=True)
