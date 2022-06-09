# Gensim Core Concepts - Exploration
As per [here](https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html) though with different sample data and a few tweaks.

## Work Breakdown Structure:
### High Priority:
* Add computer science/programming controlled vocabularies. #DONE
* Parse software engineering sample_corpus with gensim utilities. #FOCUS
* Design ontology model for software engineering domain. #TODO
    * Include marker for single/few-letter domain words (i.e. "c", "R", etc.)
    * Identify relevance and utility of tags present in `stackexchange_tags.tag_description` (surrounded by square brackets). 
        * Can a reliable relationship be established between them and the parent tag?
* Build modelled controlled vocabulary for software engineering domain. #TODO
    

### Lower Priority:
* Design mechanism to disambiguate single/few-letter domain words from non-domain instances. #TODO

<style>
todo { background-color: Yellow; color: SteelBlue }
recurrent { background-color: Gold; color: SteelBlue }
refactor { background-color: SpringGreen; color: DarkGreen }
done { background-color: Green; color: PaleGreen }
test { background-color: Coral; color: DarkRed }
fixme { background-color: Crimson; color: white }
focus { background-color: DeepSkyBlue; color: MediumBlue }
reqspec { background-color: MediumPurple; color: Indigo }
doubt { background-color: #FF00FF; color: Yellow }
wait { background-color: Pink; color: Crimson }
blocked { background-color: Crimson; color: Yellow }
</style>
