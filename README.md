# Multimodal disinformation detection on Fakeddit

In online social networks, the dissemination of disinformation has serious political, social and economic consequences. Analyzing posts is challenging because they are manifested through heterogeneous information modalities such as text or image. The rapid proliferation of this heterogeneous information underscores the critical need for automated detection methods. In this paper, we propose a multimodal architecture capable of identifying manipulated posts. Our model is trained with a large scale multimodal dataset that contains not only image and caption, but also comments and metadata for each post. We propose a coding of the comments that helps not only to gather the semantics of the comments but also their tree structure. A deep learning early fusion technique is adopted using CLIP as a pretrained encoder. Hidden representations are combined according to the information channel, so that multimodal and unimodal representations are processed. The results tested against the Fakeddit dataset show that several information channels sharing the same target can provide more information than separate ones. We achieve a very competent performance in the 2-way label task with significant limitations (0.9506 acc_test) and overpass the state-of-the-art in the 3-way and 6-way label tasks with 0.9509 acc_test and 0.9371 acc_test respectively.

In order to replicate the experiments you need:

1) Download Fakeddit dataset (https://fakeddit.netlify.app/) and save the .tsv files in the path ./dataset and the images in the path ./dataset/{subset}.
2) Adjust the main function playing with the ./data_builder.py and the ./architecture_builder.py libraries. Note: there is no console commands implemented yet.

