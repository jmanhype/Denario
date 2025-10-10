from denario import Denario, Journal

folder = "GW231123"
astro_pilot = Denario(project_dir=folder)

astro_pilot.set_data_description(f"{folder}/input.md")

astro_pilot.get_idea_fast(llm='gpt-4.1-mini') 

astro_pilot.check_idea_fast(llm='gpt-4.1-mini', max_iterations=7) 

astro_pilot.get_method_fast(llm='gpt-4.1-mini') 

astro_pilot.get_results(engineer_model='gpt-4.1-mini',
                        researcher_model='gpt-4.1-mini',
                        planner_model='gpt-4.1-mini',
                        plan_reviewer_model='gpt-4.1-mini',
                        default_orchestration_model='gpt-4.1-mini',
                        default_formatter_model='gpt-5-mini',
                        )
                        
astro_pilot.get_paper(journal=Journal.AAS, llm='gpt-4.1-mini', add_citations=False) 

astro_pilot.referee(llm='gpt-4.1-mini')
