# standard packages
import os
import re
import json
import argparse
import itertools
from pathlib import Path

# less standard packages
import networkx as nx
import spacy
import matplotlib.pyplot as plt


def load_data(data_path: Path):
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
        f.close()
    return text


def load_json(data_path: Path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def remove_gutenberg(text: str):
    # it looks like they are signaled by "***" to start and end, so can we just key off that? Probably not the best but not horrible.
    # *** occurs 4 times, 2 before text and 2 after text
    # and let's also get rid of all the new line chars
    return text.split("***")[2]


def initialize_model(data_folder: Path):
    # initialize the spacy model
    # TODO: uncomment this before submitting - we don't need to download every time while running
    # spacy.cli.download("en_core_web_sm")
    # and make sure to disable named entity recognition - we only want the characters we will specify
    model = spacy.load("en_core_web_sm", disable=["ner"])

    # load all the characters that we care about
    char_path = Path(data_folder, "characters.json")
    char_patterns = load_json(char_path)

    # and add them to the spacy model so that spacy only cares about these characters
    ruler = model.add_pipe("entity_ruler")
    ruler.add_patterns(char_patterns)
    return model


def add_to_character_network(network: nx.Graph, chars_by_segment: list):

    for chars_segment in chars_by_segment:
        # first, remove any characters that are mentioned more than once
        single_char_mention = list(set(chars_segment))

        # now iterate through all possible pairs of characters and add/increment their weight
        # ChatGPT showed me an example of how to do this easily, which I paraphrased from
        for character_1, character_2 in itertools.combinations(single_char_mention, 2):
            if network.has_edge(character_1, character_2):
                # if there is already an edge between these characters, increment its weight
                network[character_1][character_2]["weight"] += 1
            else:
                # this is a new character combination, create an edge
                network.add_edge(character_1, character_2, weight=1)
    return network


def get_characters(document: str, model: spacy.Language):

    # load a spaCy model
    # give spaCy the doc:
    result = model(document)

    # and now pull out all the characters
    characters = []
    for entry in result.ents:
        # ChatGPT helped me with the syntax here. Seems a bit ugly though. Wonder if the spaCy API
        # doc shows a cleaner way to do this...
        if entry.label_ == "PERSON":

            character = entry.ent_id_
            characters.append(character)
    return characters


def count_character_occurrences(characters_by_segment: list, data_folder: Path, save_id: int):

    # let's count the character occurrence frequency, key by id
    character_counts = {}
    for char_doc in characters_by_segment:
        for c_id in char_doc:
            try:
                character_counts[c_id] += 1
            except KeyError:
                # this is a new character
                character_counts[c_id] = 1

    # now sort by counts
    sorted_characters = sorted(character_counts.items(), key=lambda i: i[1], reverse=True)

    # and dump these out to a json file? - so we can inspect for debugging
    characters_file = Path(data_folder, f"character_counts_{save_id}.json")
    normalized_counts_dict = dict(sorted_characters)
    with open(characters_file, "w") as f:
        json.dump(normalized_counts_dict, f, indent=4)
        f.close()
    return normalized_counts_dict


def split_into_sentences(text):
    # Copied from my HW6 submission
    # CITATION: Gemini assistance provided here
    # This regex attempts to split sentences based on common punctuation marks
    # (. ? !) followed by whitespace, while trying to avoid splitting within abbreviations.
    # It looks for a sentence-ending punctuation mark, followed by whitespace,
    # and uses a negative lookbehind to prevent splitting after common abbreviations
    # like "Mr.", "Dr.", "etc." or initials like "A.B.C."
    sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s'
    sentences = re.split(sentence_endings, text)
    # Filter out empty strings that might result from the split
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def plot_network(net: nx.Graph, save_name: os.PathLike, title: str):
    # plotting brought to you with help from Gemini
    plt.clf()
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(net, k=0.3)

    # draw the nodes and edges
    nx.draw_networkx_nodes(net, pos, node_color='violet', node_size=100)
    nx.draw_networkx_edges(net, pos, edge_color='gray', width=2)

    nx.draw_networkx_labels(net, pos, font_color='darkblue', font_size=10, font_weight='bold')
    plt.title(title)
    plt.savefig(save_name)


def clean_and_parse_text(text: str, data_folder: Path):
    # basic cleaning and parse into sentences
    gutenberg_removed = remove_gutenberg(text)
    # remove all the newline characters (might not need to do this, but its caused problems in the past...)
    newline_removed = gutenberg_removed.replace("\n", " ")
    # and lower case everything
    lower_cased = newline_removed.lower()

    # now split the novel up by chapter
    chapter_path = Path(data_folder, "chapter_keys.json")
    chapter_keys = load_json(chapter_path)

    chapters = []
    if chapter_keys["manual"]:
        # handle manual chapter splitting
        for i, split_key in enumerate(chapter_keys["keys"]):
            chap_start = lower_cased.split(split_key)[-1]
            if i != len(chapter_keys["keys"]) - 1:
                chap = chap_start.split(chapter_keys["keys"][i + 1])[0]
            else:
                # last chapter!
                chap = chap_start
            chapters.append(chap)
    else:
        # sometimes the chapter pattern is not the same across all chapters. For example, some novels have a prologue,
        # or an epilogue, or something else entirely. So we manually specify the start and end splits
        begin_split = chapter_keys["begin_key"]
        main_split = chapter_keys["main_key"]
        end_split = chapter_keys["end_key"]
        if begin_split is not None:
            # we want everything after the FIRST beginning split (should generally occur once, but just be safe in case it doesn't)
            chapters_text = " ".join(lower_cased.split(begin_split)[1:])
        else:
            raise ValueError("Chapter keys must define a beginning key!")

        # next grab the end split - this can be none
        last_chapter = ""
        if end_split is not None:
            last_chapter = chapters_text.split(end_split)[-1]

        # finally grab all the normal chapters
        if main_split is not None:
            chapters = chapters_text.split(main_split)
        else:
            raise ValueError("Chapter keys must define a main chapter key!")

        # now add on the last chapter, if we had to manually grab it
        if last_chapter:
            chapters.append(last_chapter)
    print(f"There are {len(chapters)} chapters in '{data_folder.stem.replace('_', ' ')}'")
    return chapters


def plot_timeseries(time_series: dict, chapters: list, role_keys: dict, data_folder: Path, y_label: str):
    # plot degree centrality
    plt.clf()
    # first plot all the other characters
    other_labeled = False
    other_characters = [char for char in time_series.keys() if
                        char not in role_keys["victims"] and char not in role_keys["culprits"] and char not in role_keys["innocent"]]
    for other_char in other_characters:
        plt.plot(chapters, time_series[other_char], color="dodgerblue",
                 label="Suspect" if not other_labeled else None, alpha=0.7)
        other_labeled = True

    # plot the known innocent
    innocent_labeled = False
    innocent_characters = [char for char in time_series.keys() if
                           char in role_keys["innocent"]]
    for in_char in innocent_characters:
        plt.plot(chapters, time_series[in_char], color="teal",
                 label="Innocent" if not innocent_labeled else None, alpha=0.7)
        innocent_labeled = True

    # now plot the victims
    victim_labeled = False
    victims = [char for char in time_series.keys() if char in role_keys["victims"]]
    for v in victims:
        plt.plot(chapters, time_series[v], color="salmon",
                 label="Victim" if not victim_labeled else None)
        victim_labeled = True

    # and finally plot the culprit
    culprit_labeled = False
    culprits = [char for char in time_series.keys() if char in role_keys["culprits"]]
    for c in culprits:
        plt.plot(chapters, time_series[c], color="purple",
                 label="Culprit" if not culprit_labeled else None)
        culprit_labeled = True

    plt.title(f"{data_folder.stem.replace('_', ' ')}")
    plt.xlabel("Chapter")
    plt.ylabel(y_label)
    plt.legend()
    centrality_plot_path = Path(data_folder, f"{y_label.lower().replace(' ', '_')}.png")
    plt.savefig(centrality_plot_path) #, dpi=100)


def main(data_folder: Path):
    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"Folder {data_folder} does not exist.")

    # the text data should match the folder name, with a .txt extension
    dpath = Path(data_folder, f"{data_folder.stem}.txt")
    loaded_text = load_data(dpath)

    # clean and parse into chapters
    chapters = clean_and_parse_text(loaded_text, data_folder)

    # and make a chapters folder, if it doesn't already exist. We'll dump plots in here
    chapters_folder = Path(data_folder, "chapters")
    if not os.path.isdir(chapters_folder):
        print(f"Making new dir: {chapters_folder}")
        os.mkdir(chapters_folder)

    # make a folder for degree centrality plots if it doesn't already exists
    deg_cent_folder = Path(chapters_folder, "degree_centrality")
    if not os.path.exists(deg_cent_folder):
        print(f"Making new folder: {deg_cent_folder}")
        os.mkdir(deg_cent_folder)
    # make a folder for character counts plots if it doesn't already exists
    counts_folder = Path(chapters_folder, "character_counts")
    if not os.path.exists(counts_folder):
        print(f"Making new folder: {counts_folder}")
        os.mkdir(counts_folder)

    # Create two empty dictionaries to store the results of each chapter in
    cumulative_centrality = {}
    cumulative_counts = {}

    # and we'll need to track the number of character mentions
    sum_char_mentions = 0

    # initialize a network
    n = nx.Graph()

    # and initialize a spacy model
    model = initialize_model(data_folder)

    # now build the networks on a chapter basis
    for i, c in enumerate(chapters):

        print(f"Processing chapter: {i}")

        all_sentences = split_into_sentences(c)

        chars_by_sentence = []
        for s in all_sentences:

            chars = get_characters(s, model)
            # be sure to drop all sentences without any characters altogether
            if len(chars) != 0:
                chars_by_sentence.append(chars)

        # count the mentions of each character
        # format the index nicely so sorting can be done later
        str_i = str(i) if i > 9 else f"0{i}"
        char_counts = count_character_occurrences(chars_by_sentence, counts_folder, str_i)
        # last thing - normalize the counts so that we can compare across different segments/novels
        # and round a bit to keep things pretty
        sum_char_mentions = sum(char_counts.values()) + sum_char_mentions
        normalized_counts = dict([(i, round(j / sum_char_mentions, 4)) for i, j in char_counts.items()])

        # now add the counts to the cumulative counts
        # first we need a set of a unique keys from both
        unique_chars = set(list(cumulative_counts.keys()) + list(normalized_counts.keys()))
        for key in unique_chars:
            try:
                new_count = normalized_counts[key]
            except KeyError:
                # this character was not mentioned in this chapter
                # but it must already be in the character counts
                # so we can just add the last value
                new_count = cumulative_counts[key][-1]
                cumulative_counts[key].append(new_count)
            else:
                # this character was mentioned in this chapter, but may not yet exist
                try:
                    cumulative_counts[key].append(new_count)
                except KeyError:
                    # this character does not yet exist in the cumulative counts!
                    zeroes = [0.0] * i  # note that i is zero indexed, so this builds correctly up until this index
                    cumulative_counts[key] = zeroes + [new_count]

        # we need the characters by sentence, but only with their ids
        char_net = add_to_character_network(n, chars_by_sentence)
        degree_centrality = nx.degree_centrality(char_net)

        # now sort by largest
        sorted_deg_cen = dict(sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True))

        # let's round off nicely before displaying
        sorted_deg_cen_round = {k: round(v, 2) for (k, v) in sorted_deg_cen.items()}

        save_centrality_path = Path(deg_cent_folder, f"character_degree_centrality_{str_i}.json")
        deg_cen_dict = dict(sorted_deg_cen_round)
        with open(save_centrality_path, "w") as f:
            json.dump(deg_cen_dict, f, indent=4)
            f.close()

        save_path = Path(chapters_folder, f"character_network_{str_i}.png")
        plot_network(char_net, save_path, data_folder.stem.replace("_", " "))

        # now add to the cumulative dictionaries. Note that not all characters will be present in each chapter!
        for k, v in deg_cen_dict.items():
            try:
                # if this character already exists, just add to it's existing list
                cumulative_centrality[k].append(v)
            except KeyError:
                # this character does not yet exist. Fill in 0s for all previous chapters
                zeroes = [0.0] * i  # note that i is zero indexed, so this builds correctly up until this index
                cumulative_centrality[k] = zeroes + [v]

    # wahoo, now we can plot the cumulative data!
    # but first we need the character keys for color coding
    role_path = Path(data_folder, "role_key.json")
    role_keys = load_json(role_path)

    chapters_to_plot = list(range(len(chapters)))

    plot_timeseries(cumulative_centrality, chapters_to_plot, role_keys, data_folder, "Cumulative Degree Centrality")
    plot_timeseries(cumulative_counts, chapters_to_plot, role_keys, data_folder, "Cumulative Character Mention Frequency")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Character Network Builder")
    parser.add_argument("--data_folder",
                        type=str,
                        required=True,
                        help="Path to the folder containing novel text data.")

    args = parser.parse_args()
    main(Path(args.data_folder))
