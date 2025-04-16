from transformers import EsmModel, EsmConfig


def genTSNE():
    device = 'cuda'

    df = pd.read_csv(csv)
    test_embeddings_h5_file = h5py.File(path_test_embeddings, "r")

    print("Encoding all retrieval sequences to plot onto t_SNE")
    all_embeddings = []
    all_embeddings_pre_model = []
    all_label_color = []
    for _, retrieval_row in df.iterrows():
        # Get the embed sequence from the h5 file
        retrieval_sequence = retrieval_row['sequence']
        retrieval_label = retrieval_row['label']
        retrieval_ESM_embed = torch.tensor(np.array(test_embeddings_h5_file[f'{retrieval_sequence}/embedding']), dtype=torch.float32)
        retrieval_ESM_embed_atten_mask = torch.tensor(np.array(test_embeddings_h5_file[f'{retrieval_sequence}/atten_mask']), dtype=torch.float32)
        
        retrieval_ESM_embed_plot = torch.squeeze(retrieval_ESM_embed, 0)
        retrieval_ESM_embed_plot = retrieval_ESM_embed_plot.mean(dim=0)
        print(retrieval_ESM_embed_plot.shape)
        retrieval_ESM_embed_plot = retrieval_ESM_embed_plot.detach().numpy()
        all_embeddings_pre_model.append(retrieval_ESM_embed_plot)

        retrieval_ESM_embed = retrieval_ESM_embed.to(device)
        retrieval_ESM_embed_atten_mask = retrieval_ESM_embed_atten_mask.to(device)

        retrieval_ESM_embed_atten_mask = retrieval_ESM_embed_atten_mask.T

        # Feed it into model
        retrieval_seq_output_embedding = model(retrieval_ESM_embed,src_key_padding_mask=retrieval_ESM_embed_atten_mask, src_mask=None)

        #print(f"Output shape = {retrieval_seq_output_embedding.shape}")
        #print(type(retrieval_seq_output_embedding))

        #vector = np.array([retrieval_seq_output_embedding.cpu()], dtype=np.float32)
        retrieval_seq_output_embedding = torch.squeeze(retrieval_seq_output_embedding, 0)
        vector = retrieval_seq_output_embedding.detach().cpu().numpy()
        all_embeddings.append(vector)

        if retrieval_label == 1:
            all_label_color.append("blue")
        else:
            all_label_color.append("red")

    #convert to array 
    all_embeddings = np.array(all_embeddings)

    print(all_embeddings.shape)

    tsne = TSNE(n_components=2, random_state=42)
    transformed_vectors = tsne.fit_transform(all_embeddings)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.scatter(transformed_vectors[:, 0], transformed_vectors[:, 1], c=all_label_color, alpha=0.7)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE Visualization Post-Model")
    plt.savefig("tSNE_plot_post")
    

    all_embeddings_pre_model = np.array(all_embeddings_pre_model)
    transformed_vectors_pre = tsne.fit_transform(all_embeddings_pre_model)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.scatter(transformed_vectors_pre[:, 0], transformed_vectors_pre[:, 1], c=all_label_color, alpha=0.7)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE Visualization Pre-Model")
    plt.savefig("tSNE_plot_pre")