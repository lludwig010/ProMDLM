import pandas as pd


if __name__ == "__main__":

    full_files = ["/home/jtso3/ghassan/ProMDLM/generated_sequences/lysozyme_100_test_set_final_results_full.csv",
    "/home/jtso3/ghassan/ProMDLM/generated_sequences/generated_sequences_two_stage_results_full.csv",
    "/home/jtso3/ghassan/ProMDLM/generated_sequences/generated_sequences_progen_results_full.csv",
    "/home/jtso3/ghassan/ProMDLM/generated_sequences/generated_sequences_increment_results_full.csv",
    "/home/jtso3/ghassan/ProMDLM/generated_sequences/generated_sequences_fulldiff_results_full.csv"]

    TEMPERATURE=1
    

    filtered_files =[]
    percent_passed = []
    for path in full_files:
        # Load the dataframe from the CSV file
        df = pd.read_csv(path)
        
        # keep rows with temp 1
        filtered_df = df[((df["temperature"]==TEMPERATURE)|(df["temperature"].isna()))]
        len_before_ppl_filter = len(filtered_df)
        # Filter out rows where the sequence length is shorter than 50 and the entropy is lower than 3.5
        filtered_df = filtered_df[(filtered_df["sequence"].str.len() >= 50) & (filtered_df["entropy"] >= 3.5)]
        
        # Print the number of lines left
        print(f"{path}: {len(filtered_df)} lines left after filtering.")
        # Calculate the percentage of sequences that passed the filter
        percent_passed.append(len(filtered_df) / len_before_ppl_filter * 100)
        
        # Save the filtered dataframe to a new CSV file
        filtered_path = path.replace(".csv", f"_t{TEMPERATURE}_filtered.csv")
        filtered_df.to_csv(filtered_path, index=False)
        filtered_files.append(filtered_path)