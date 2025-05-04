import llama_cpp._internals as _internals
_internals.LlamaSampler.__del__ = lambda self: None

from llama_cpp import Llama
from transformers import pipeline
import re
from time import sleep

def initialize_models():
    """Initialize the LLM and summarization models"""
    llm = Llama(
        model_path="/Users/avneh/llama-models/mistral-7b/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        n_ctx=32768,
        n_threads=6,
        n_gpu_layers=35,
        chat_format="mistral-instruct",
        verbose=False,
        stop=None
    )
    
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return llm, summarizer

def offline_summarize(text, summarizer):
    """
    Summarize the provided text using Hugging Face BART model.
    Returns the summarized version of the input text.
    """
    max_input_length = 1024
    if len(text.split()) > max_input_length:
        text = text[:max_input_length]
    
    summary = summarizer(text, max_length=50, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def split_into_chunks(input_string, chunk_size=467):
    """Split input text into chunks of approximately equal size"""
    sentences = re.split(r'(?<=[.!?]) +', input_string)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length > chunk_size:
            if current_chunk:
                chunks.append({'chunk': len(chunks), 'text': ' '.join(current_chunk)})
                current_chunk = []
                current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append({'chunk': len(chunks), 'text': ' '.join(current_chunk)})

    return chunks

def process_chunk(chunk, previous_summary, llm, summarizer, explanation_prompt="Explain what this means in simple terms"):
    """Process a single chunk of text and return its explanation and summary"""
    if chunk['chunk'] == 0:
        explanation_prompt_text = f"[INST] {explanation_prompt}: {chunk['text']} [/INST]"
    else:
        explanation_prompt_text = f"[INST] Previous context summary: {explanation_prompt}: {previous_summary}, considering the previous context, fulfill the following prompt: {explanation_prompt}: {chunk['text']} [/INST]"
    
    prompt_length = len(explanation_prompt_text.split())
    print(f"Prompt length for chunk {chunk['chunk']}: {prompt_length} words")
    
    if prompt_length > 600:
        print(f"Prompt too long for chunk {chunk['chunk']}, skipping...")
        return None, previous_summary
    
    try:
        output = llm(explanation_prompt_text, max_tokens=1024, temperature=0.1)
        explanation = output["choices"][0]["text"].strip()
        
        if not explanation:
            explanation = "No explanation returned."
        
        print(f"Chunk {chunk['chunk']} Explanation: {explanation}\n")

        if explanation and explanation != "No explanation returned.":
            new_summary = offline_summarize(explanation, summarizer)
        else:
            new_summary = "Previous explanation was empty or errored."
            
    except Exception as e:
        print(f"Error processing chunk {chunk['chunk']}: {e}")
        explanation = "Error during explanation."
        new_summary = "Error occurred in previous chunk processing."

    return explanation, new_summary

def process_text(input_text, llm, summarizer, explanation_prompt="Explain what this means in simple terms"):
    """Process the entire input text through chunks"""
    chunks = split_into_chunks(input_text)
    previous_summary = ""
    
    for chunk in chunks:
        explanation, previous_summary = process_chunk(chunk, previous_summary, llm, summarizer, explanation_prompt)
        sleep(0.7)  # Delay to avoid rate limiting

def main():
    """Main function to run the text processing pipeline"""
    # Example text
    input_text = """The Golden State Warriors, once the NBA's undisputed dynasty, find themselves at a fascinating crossroads in the 2023-24 season. After a decade of dominance that included four championships and six Finals appearances, the team is navigating the delicate balance between maintaining their championship aspirations and managing an aging core while developing their next generation of stars.

    At the heart of the Warriors' current situation is the continued excellence of Stephen Curry, who remains one of the league's most impactful players despite being 35 years old. Curry's shooting prowess and leadership continue to anchor the team, as he averages over 27 points per game while maintaining his trademark efficiency from beyond the arc. His ability to create space and generate offense remains unparalleled, and he's shown no signs of significant decline in his game.

    However, the supporting cast around Curry has undergone significant changes. Klay Thompson, once considered one of the league's premier two-way players, has struggled to regain his pre-injury form. While he still shows flashes of his old brilliance, his defensive mobility and consistency have diminished. The Warriors have had to adjust their defensive schemes to account for this, often hiding Thompson on less dynamic offensive players.

    Draymond Green's role has evolved as well. While still the team's emotional leader and defensive anchor, his offensive contributions have become more selective. The Warriors have increasingly relied on his playmaking and basketball IQ rather than his scoring. Green's recent suspension for on-court altercations has also raised questions about his ability to maintain composure in high-pressure situations.

    The development of Jonathan Kuminga and Moses Moody has been a bright spot for the Warriors. Kuminga, in particular, has shown significant growth, demonstrating improved shooting and defensive versatility. His athleticism and ability to attack the rim have provided the Warriors with a much-needed dimension they've lacked in recent years. Moody's development has been more gradual, but he's shown promise as a reliable three-and-D wing.

    The Warriors' front office made a significant move in the offseason by acquiring Chris Paul, a future Hall of Fame point guard. While Paul's addition was initially met with skepticism due to his age and injury history, he's proven to be a valuable addition. His leadership and ability to run the second unit have stabilized the Warriors' bench, which was a significant weakness last season. Paul's presence has also allowed Curry to play more off-ball, preserving his energy for crucial moments.

    The center position remains a question mark for the Warriors. Kevon Looney continues to be a reliable presence, but his limited offensive game can be exploited in certain matchups. The team has experimented with various combinations, including small-ball lineups featuring Green at center, but this approach has its limitations against teams with dominant big men.

    Steve Kerr's coaching has been tested this season as he balances the need to win now with developing young talent. His ability to manage egos and maintain team chemistry has been crucial, especially during periods of inconsistent play. The Warriors' motion offense remains effective, but teams have become better at defending it, forcing Kerr to implement more creative sets and counters.

    The Western Conference has become increasingly competitive, with several teams making significant improvements. The Warriors find themselves in a tight race for playoff positioning, with every game carrying significant weight. Their ability to secure home-court advantage could be crucial in the postseason, given their strong home record at Chase Center.

    Financially, the Warriors continue to operate with the league's highest payroll, thanks to their success and the revenue generated by their new arena. However, the new CBA's stricter luxury tax penalties have forced the team to be more strategic with their roster construction. The front office faces difficult decisions regarding contract extensions for their young players while maintaining flexibility for potential roster upgrades.

    The Warriors' fan base remains one of the most passionate in the league, but expectations have evolved. While championship aspirations remain, there's a growing acceptance of the need for a transitional period. The team's ability to remain competitive while developing their next core has been impressive, but the window for another championship with the current group is undoubtedly narrowing.

    Looking ahead, the Warriors face several key questions. Can they make one more deep playoff run with their current core? How quickly can their young players develop into reliable contributors? And perhaps most importantly, how will they manage the eventual transition from the Curry-Thompson-Green era to the next generation of Warriors basketball?

    The 2023-24 season represents a critical juncture for the franchise. While they may not be the overwhelming favorites they once were, the Warriors remain a dangerous team capable of beating anyone on any given night. Their combination of championship experience, elite talent, and emerging young players makes them one of the most fascinating teams to watch as the season progresses.

    The Warriors' current state is one of transition and adaptation. They're no longer the juggernaut that dominated the league for nearly a decade, but they're far from being a rebuilding team. Instead, they're navigating the challenging middle ground between maintaining their championship pedigree and preparing for the future. How they manage this balance will determine their success not just this season, but for years to come."""
    
    # Initialize models
    llm, summarizer = initialize_models()
    
    try:
        # Process the text with a custom explanation prompt
        explanation_prompt = "How would this affect the United States gold exchange rate?"
        process_text(input_text, llm, summarizer, explanation_prompt)
    finally:
        # Clean up
        llm.close()

if __name__ == "__main__":
    main()
