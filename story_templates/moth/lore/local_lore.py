# story_templates/moth/lore/local_lore.py
"""
Local lore - myths, histories, landmarks, and cultural elements for SF Bay Area
"""

from typing import Dict, Any, List

class SFLocalLore:
    """Local myths, histories, and cultural elements for SF Bay Area"""
    
    @staticmethod
    def get_specific_locations() -> List[Dict[str, Any]]:
        """Get story-specific detailed locations"""
        return [
            {
                "name": "The Rose Garden Café",
                "address": "Valencia Street between 20th and 21st",
                "type": "recruitment_hub",
                "description": (
                    "A perfectly normal Mission café that serves exceptional lavender lattes "
                    "and rose petal scones. Local art rotates monthly, always featuring "
                    "strong women in subtle positions of power. The owner, Lily Chen, "
                    "remembers everyone's order and occasionally their secrets. Tuesday "
                    "book clubs read Virginia Woolf but discuss power dynamics. The back "
                    "room hosts 'wine tastings' where the pairing notes include dominance "
                    "and submission."
                ),
                "public_function": "Trendy neighborhood café",
                "hidden_function": "Entry point to the Thorn Garden",
                "atmosphere": {
                    "Day": "Laptops and casual meetings",
                    "Evening": "Poetry readings with hidden meanings",
                    "Late night": "Private gatherings by invitation"
                },
                "recognition_signs": [
                    "Rose imagery in subtle places",
                    "Staff who observe more than orders",
                    "Customers who bow heads slightly to Lily",
                    "Books left on specific tables as messages"
                ],
                "special_features": {
                    "The reading nook": "Where initial assessments happen",
                    "The rose wall": "Instagram backdrop and message board",
                    "The private room": "Wine tastings that transform perspectives"
                }
            },
            {
                "name": "Thornfield & Associates",
                "address": "555 California Street, 40th Floor",
                "type": "power_brokerage",
                "description": (
                    "Elite law firm specializing in family law, estate planning, and 'special "
                    "arrangements.' The reception area whispers old money while the corner "
                    "offices scream new power. Senior partners meet in a conference room "
                    "with no windows and exceptional soundproofing. Their client agreements "
                    "include NDAs that would terrify the uninitiated. Some executives leave "
                    "meetings walking differently than they entered."
                ),
                "public_function": "White-shoe law firm",
                "hidden_function": "Legal architecture for the shadow matriarchy",
                "key_figures": {
                    "Evelyn Thornfield": "Managing Partner with interesting hobbies",
                    "Sarah Kim": "Senior Associate who specializes in behavior",
                    "The Advisory Board": "Seven women who shape more than law"
                },
                "services": {
                    "Official": "Prenups, trusts, corporate structure",
                    "Unofficial": "Behavioral contracts, submission agreements, silence pacts"
                },
                "client_signs": "Powerful men who defer to female associates"
            },
            {
                "name": "The Wellness Collective",
                "address": "Former church on Divisadero",
                "type": "transformation_center",
                "description": (
                    "A converted Gothic church now hosting yoga, meditation, and 'healing "
                    "circles.' The stained glass depicts feminine divine figures. Sound "
                    "baths happen in the former sanctuary where acoustics carry whispers "
                    "to unexpected ears. The basement, once church storage, now contains "
                    "equipment for more intense transformations. Members progress through "
                    "levels of enlightenment that correlate to depths of submission."
                ),
                "public_programs": [
                    "Vinyasa flow classes",
                    "Women's empowerment workshops",
                    "Executive stress relief programs"
                ],
                "deeper_work": [
                    "Power exchange as therapy",
                    "Behavioral modification through bodywork",
                    "Corporate leadership redefinition"
                ],
                "architectural_features": {
                    "The sanctuary": "Public classes with private undertones",
                    "The confessionals": "Repurposed for one-on-one sessions",
                    "The basement": "Equipment not found in typical yoga studios"
                },
                "membership_levels": "Curious, Committed, Converted, Owned"
            },
            {
                "name": "Montenegro Modern Gallery",
                "address": "Jackson Square Historic District",
                "type": "cultural_conversion",
                "description": (
                    "High-end gallery in a 1850s building with very modern purposes. "
                    "Isabella Montenegro curates art that reveals the viewer more than "
                    "the viewed. Tech executives buy aggressive pieces that betray their "
                    "need to submit. Opening nights are performance art where power "
                    "dynamics play out over champagne. The climate-controlled storage "
                    "rooms preserve more than paintings."
                ),
                "exhibitions": {
                    "Public": "Contemporary feminist art",
                    "Private viewings": "Pieces that inspire submission",
                    "Storage collection": "Art that transforms viewers"
                },
                "clientele": [
                    "New money seeking culture",
                    "Old money knowing secrets",
                    "Those referred by roses"
                ],
                "transformation_method": "Art as mirror, purchase as admission",
                "famous_pieces": {
                    "'Thorns Ascendant'": "Viewers report life changes",
                    "'The Yielding'": "Bought by three CEOs who resigned after",
                    "'Rose Imperious'": "Not for sale, changes location nightly"
                }
            },
            {
                "name": "Maison Noir",
                "address": "Undisclosed, SoMa district",
                "type": "finishing_school",
                "description": (
                    "A black door in an alley off Folsom. No sign, no number, but those "
                    "who need it find it. Inside, a Victorian brothel aesthetic meets "
                    "modern dungeon efficiency. But this isn't sex work - it's power work. "
                    "Dominants train in psychological control, submissives learn to serve "
                    "power. The graduation ceremonies involve contracts that bind more than "
                    "bodies."
                ),
                "curriculum": {
                    "Basic": "Protocol, safety, consent frameworks",
                    "Intermediate": "Psychological manipulation, behavioral reading",
                    "Advanced": "Life control, identity modification",
                    "Masters": "Creating permanent change"
                },
                "faculty": "Mistresses who've broken captains of industry",
                "alumni_network": "Places graduates throughout Bay Area power structures",
                "admission": "Referral only, extensive vetting",
                "graduation_rate": "60% - others find they're meant to serve"
            },
            {
                "name": "The Butterfly House",
                "address": "Marina Boulevard (appears as private residence)",
                "type": "chrysalis_safehouse",
                "description": (
                    "Mediterranean villa overlooking the bay where transformations complete. "
                    "Those who enter as victims emerge as victors. The therapeutic program "
                    "includes traditional trauma work and power reclamation through dominance "
                    "training. The garden grows herbs that heal and plants that punish. At "
                    "night, lights in windows signal safety or summons."
                ),
                "public_face": "Executive women's retreat",
                "true_purpose": "Turning prey into predators (ethical ones)",
                "programs": [
                    "Trauma recovery through power reclamation",
                    "Identity reconstruction services",
                    "Dominance training for the formerly submissive",
                    "Protection network integration"
                ],
                "security": {
                    "Visible": "High walls, security company",
                    "Hidden": "Underground exits, safe rooms, armed gardeners"
                },
                "success_stories": "Former victims now running Fortune 500 companies"
            },
            {
                "name": "The Thorn Archive",
                "address": "Climate-controlled facility, Peninsula",
                "type": "information_repository",
                "description": (
                    "Looks like a wine storage facility, functions as the memory of the "
                    "matriarchy. Every video, every contract, every confession stored in "
                    "redundant systems. The Queen of Thorns (or her librarians) can access "
                    "decades of leverage. Some executives pay millions just to know what's "
                    "in their file."
                ),
                "contents": [
                    "Compromising materials on half of Silicon Valley",
                    "Historical records of the movement",
                    "Transformation documentation",
                    "The unedited histories"
                ],
                "access_levels": "Need to know basis, even for roses",
                "security": "Biometric, quantum encryption, and ladies who lunch with guns",
                "urban_legend": "Contains the true identity of the first Queen"
            },
            {
                "name": "Dr. Johnson's Pacific Heights Practice",
                "address": "Victorian row house near Lafayette Park",
                "type": "psychological_restructuring",
                "description": (
                    "Psychiatrist office that looks like old San Francisco money. Oriental "
                    "rugs, leather chairs, and therapeutic methods not found in the DSM. "
                    "Dr. Amara Johnson helps willing clients discover their need to submit, "
                    "guides unwilling ones toward acceptance. The pharmaceuticals she "
                    "prescribes aren't all FDA approved. Some patients emerge with new "
                    "orientations toward authority."
                ),
                "therapeutic_approaches": [
                    "Traditional talk therapy (for show)",
                    "Hypnotherapy with permanent suggestions",
                    "Pharmaceutical assistance for resistance",
                    "Behavioral modification through guided experience"
                ],
                "client_base": "Referrals from the Garden, court-ordered alternatives",
                "success_metric": "Recidivism rate: 0%",
                "waiting_room_art": "Rothko prints that induce contemplation of surrender"
            },
            {
                "name": "The Folsom Street Foundry",
                "address": "Industrial building off 11th Street",
                "type": "equipment_and_education",
                "description": (
                    "Metalworking shop that produces more than art. Custom restraints that "
                    "double as jewelry, furniture that transforms spaces and people. The "
                    "welding classes teach precision that applies beyond metalwork. The "
                    "showroom displays pieces that make viewers imagine themselves in them. "
                    "Special orders require measurements beyond the physical."
                ),
                "public_products": [
                    "Industrial art pieces",
                    "Custom furniture",
                    "Architectural elements"
                ],
                "special_orders": [
                    "Restraint systems disguised as decor",
                    "Furniture for specific activities",
                    "Cages that look like art installations"
                ],
                "classes_offered": {
                    "Basic welding": "Actually basic welding",
                    "Advanced fabrication": "Creating tools of transformation",
                    "Design workshop": "Psychological impact through physical form"
                },
                "clientele": "Dungeons, private collectors, people with specific needs"
            }
        ]
    
    @staticmethod
    def get_urban_myths() -> List[Dict[str, Any]]:
        """Get SF-specific urban myths for the story"""
        return [
            {
                "name": "The Executive Vanishings",
                "type": "cautionary_tale",
                "description": (
                    "They say certain executives - the ones with wandering hands and "
                    "NDAs for days - sometimes vanish from the Valley. Their LinkedIn "
                    "updates stop, their Teslas are found at SFO long-term parking, "
                    "their condos go on the market. But their funding continues, flowing "
                    "to women's organizations. Some say they've been relocated. Others "
                    "say they've been... reorganized. Rose petals in their empty offices."
                ),
                "origin_location": "Sand Hill Road",
                "spread_regions": ["Entire Bay Area tech community"],
                "believability": 7,
                "truth_level": "More true than anyone admits",
                "evidence": [
                    "Three CEOs in 2023 alone",
                    "Similar disappearance patterns",
                    "Continued financial activity",
                    "Roses at every scene"
                ],
                "official_explanation": "Burnout, moving to Austin",
                "whispered_truth": "The Garden prunes aggressively"
            },
            {
                "name": "The Consent Lottery",
                "type": "underground_legend",
                "description": (
                    "Exclusive dungeons supposedly run a lottery where winners get to "
                    "scene with the Queen of Thorns herself. Buy-in starts at $50K, "
                    "but money doesn't guarantee selection. They say she chooses based "
                    "on psychological profiles, need for correction, potential for "
                    "transformation. Winners are never the same. Losers get their money "
                    "back... minus a donation to women's shelters."
                ),
                "origin_location": "SoMa underground",
                "spread_regions": ["Financial District", "Marina", "Palo Alto"],
                "believability": 5,
                "truth_level": "Partially true - the selection process is real",
                "entry_points": "Certain gallery openings, wine auctions, charity galas",
                "selection_criteria": "Power to abuse, willingness to surrender it",
                "transformation_rate": "100% of actual participants"
            },
            {
                "name": "The BART Protector",
                "type": "guardian_legend",
                "description": (
                    "Late night BART riders tell of a woman who appears when someone's "
                    "being harassed. Well-dressed, smells of roses, speaks with quiet "
                    "authority that makes predators flee at the next stop. Security "
                    "footage mysteriously corrupts when she appears. Some say she's "
                    "the Queen herself, others say she's legion - many women sharing "
                    "a purpose and perfume."
                ),
                "origin_location": "BART system-wide",
                "spread_regions": ["Throughout Bay Area transit"],
                "believability": 8,
                "truth_level": "Multiple confirmed interventions",
                "witness_accounts": [
                    "Professional woman, 30s-50s",
                    "Designer clothes, sensible shoes",
                    "Voice that 'made him shrink'",
                    "Rose perfume lingering after"
                ],
                "BART_response": "No comment on vigilante activities"
            },
            {
                "name": "The Rose Email",
                "type": "digital_folklore",
                "description": (
                    "Screenshot collections circulate on encrypted channels - emails "
                    "containing only a rose emoji and GPS coordinates. Recipients are "
                    "always powerful men with secrets. Those who go to the coordinates "
                    "find evidence of their worst acts. Those who don't find the "
                    "evidence in their inbox Monday morning, CC'd to relevant parties. "
                    "IT security firms can't trace the source, though they've tried."
                ),
                "origin_location": "Silicon Valley C-suites",
                "spread_regions": ["Tech community worldwide"],
                "believability": 6,
                "truth_level": "Verified emails, unverified source",
                "documented_cases": 47,
                "results": [
                    "Sudden resignations",
                    "Massive anonymous donations",
                    "Public apologies without context",
                    "Behavioral transformations"
                ],
                "tech_response": "Best security can't stop them"
            },
            {
                "name": "The Monday Meeting",
                "type": "power_structure_legend",
                "description": (
                    "Every Monday at 3 PM, seven women meet somewhere in the city. "
                    "The location changes but the purpose doesn't - they decide who "
                    "rises and who falls in the Bay Area. They say board appointments, "
                    "funding rounds, and political races are settled over tea. The "
                    "Queen of Thorns either chairs the meeting or is the meeting - "
                    "seven faces of the same authority."
                ),
                "origin_location": "Pacific Heights tradition",
                "spread_regions": ["Power corridors throughout Bay Area"],
                "believability": 4,
                "truth_level": "The meetings are real, the power is debated",
                "supposed_members": [
                    "Federal judge",
                    "VC partner",
                    "Museum director",
                    "Tech executive",
                    "Society matron",
                    "Psychiatrist",
                    "Unknown seventh"
                ],
                "influence_tracking": "Monday decisions appear in Thursday news"
            },
            {
                "name": "The Transformation Spa",
                "type": "wellness_legend",
                "description": (
                    "Somewhere in Marin is a spa that offers more than hot stones "
                    "and cucumber water. Tech wives whisper about week-long retreats "
                    "that remake marriages. Husbands who enter dominant leave "
                    "submissive. The treatment list includes 'behavioral adjustment "
                    "therapy' and 'power dynamic rebalancing.' Some say it's where "
                    "the Queen trains her roses."
                ),
                "origin_location": "Marin County",
                "spread_regions": ["Wealthy enclaves"],
                "believability": 6,
                "truth_level": "Multiple locations, same purpose",
                "treatment_effects": [
                    "Reversed household dynamics",
                    "Increased charitable giving",
                    "Career changes favoring wives",
                    "Newfound enthusiasm for service"
                ],
                "booking_method": "Referral only, no public listings"
            },
            {
                "name": "The Dungeon Beneath the Pyramid",
                "type": "architectural_legend",
                "description": (
                    "The Transamerica Pyramid supposedly has sub-basements that "
                    "predate the building. During construction, they found rooms "
                    "with rings in walls, equipment that wasn't mining gear. Now "
                    "it's the city's most exclusive venue - entry requires DNA-locked "
                    "invitations and complete surrender. They say the Queen holds "
                    "court there quarterly, transforming the city's most powerful."
                ),
                "origin_location": "Financial District",
                "spread_regions": ["Executive circles"],
                "believability": 3,
                "truth_level": "Something's down there",
                "access_ritual": [
                    "Rose delivered to office",
                    "Coordinates in stem",
                    "Biometric scanner knows you",
                    "Elevator buttons that don't exist"
                ],
                "transformation_stories": "CEOs emerge as different people"
            }
        ]
    
    @staticmethod
    def get_historical_events() -> List[Dict[str, Any]]:
        """Get historical events that shape the story world"""
        return [
            {
                "event_name": "The Victorian Secret Wars",
                "date": "1890s",
                "location": "Nob Hill and Pacific Heights",
                "official_history": (
                    "Railroad barons' wives competed in charity work and social climbing."
                ),
                "hidden_history": (
                    "The city's first matriarchal power structure emerged from supposed "
                    "tea parties. Mrs. Cordelia Thornfield established 'reading circles' "
                    "that were actually power exchanges. The tradition of female authority "
                    "hiding in plain sight began with corsets and calling cards. Their "
                    "'charitable works' included reforming abusive husbands through methods "
                    "not discussed in polite society."
                ),
                "significance": 9,
                "impact": "Established the template for hidden female power",
                "legacy": "The Thornfield name still opens doors",
                "artifacts": [
                    "Diary entries in code at the History Room",
                    "Photographs with interesting power dynamics",
                    "Charity records with unusual expenditures"
                ]
            },
            {
                "event_name": "The Barbary Coast Transformation",
                "date": "1906-1917",
                "location": "Current Financial District",
                "official_history": (
                    "Red light district cleaned up after earthquake and reform movement."
                ),
                "hidden_history": (
                    "The madams didn't disappear - they evolved. The brothels became "
                    "boarding houses with interesting rules. The first Queen of Thorns "
                    "arose from these ashes, transforming exploitation into empowerment. "
                    "The tunnel systems dug for escape from raids now connect buildings "
                    "for different purposes. The architecture of control adapted."
                ),
                "significance": 10,
                "impact": "Birth of the modern shadow matriarchy",
                "legacy": "Tunnels still used, methods still practiced",
                "evidence": [
                    "Suspiciously similar disappearances of violent men",
                    "Boarding houses with no boarders but much activity",
                    "Police records with pages missing"
                ]
            },
            {
                "event_name": "The Summer of Love's Shadow",
                "date": "1967",
                "location": "Haight-Ashbury spreading citywide",
                "official_history": (
                    "Hippie movement, free love, cultural revolution."
                ),
                "hidden_history": (
                    "The 'free love' movement provided cover for deeper explorations. "
                    "Dominant women found submissive men seeking gurus. The second "
                    "Queen of Thorns emerged from a commune that was actually a training "
                    "ground. While hippies preached peace, she waged war on predators "
                    "hunting the flower children. Thirty-three saved that summer alone."
                ),
                "significance": 8,
                "impact": "Merged counterculture with shadow culture",
                "legacy": "Psychological techniques still used",
                "transformation_count": "Unknown, but Haight still whispers"
            },
            {
                "event_name": "The Moscone Assassination Aftermath",
                "date": "November 27, 1978",
                "location": "City Hall and Castro",
                "official_history": (
                    "Mayor Moscone and Harvey Milk assassinated by Dan White."
                ),
                "hidden_history": (
                    "The outrage birthed more than protests. Women who'd worked with "
                    "Milk established protection networks that evolved into enforcement "
                    "networks. The 'Twinkie Defense' inspired alternative justice systems. "
                    "They say Dan White's life in prison was... adjusted by those who "
                    "knew how to reach inside."
                ),
                "significance": 9,
                "impact": "Politicized the shadow networks",
                "legacy": "Political protection systems still active",
                "whispered_truth": "White's death wasn't suicide"
            },
            {
                "event_name": "The Loma Prieta Revelation",
                "date": "October 17, 1989",
                "location": "San Francisco Bay Area",
                "official_history": (
                    "6.9 earthquake disrupted World Series, killed 63."
                ),
                "hidden_history": (
                    "In the chaos, shadow networks proved more effective than official "
                    "ones. Women's shelters organized faster than FEMA. The Queen's "
                    "network moved through the city like angels, extracting those "
                    "trapped by more than rubble. The Marina safehouse was established "
                    "with insurance money from a mansion that 'tragically burned.'"
                ),
                "significance": 9,
                "impact": "Proved shadow systems superior in crisis",
                "legacy": "Emergency protocols still used",
                "infrastructure_born": "Three major safehouses established"
            },
            {
                "event_name": "The First Dot-Com Disruption",
                "date": "1999-2001",
                "location": "SoMa and Financial District",
                "official_history": (
                    "Tech bubble burst, fortunes lost, companies folded."
                ),
                "hidden_history": (
                    "New money met old power structures. The third Queen of Thorns "
                    "infiltrated tech culture, learning their vulnerabilities. When "
                    "the bubble burst, she acquired assets and individuals. Failed "
                    "CEOs found new purposes serving those they'd overlooked. The "
                    "money might have evaporated, but the control solidified."
                ),
                "significance": 8,
                "impact": "Integrated tech wealth into shadow systems",
                "legacy": "VC funding with strings attached",
                "conversion_rate": "23 executives transformed"
            },
            {
                "event_name": "The Twitter Revolution",
                "date": "2006-2020",
                "location": "Mid-Market and SoMa",
                "official_history": (
                    "Twitter brought tech to Mid-Market, transformed neighborhood."
                ),
                "hidden_history": (
                    "The Queen's network infiltrated from day one. As Twitter gave "
                    "everyone a voice, certain voices were amplified or silenced "
                    "through careful manipulation. The basement levels of the building "
                    "hosted more than servers. When Musk bought it, he inherited more "
                    "than code - though he hasn't discovered all the backdoors yet."
                ),
                "significance": 7,
                "impact": "Digital integration of shadow power",
                "legacy": "Control mechanisms in major platforms",
                "ongoing": "X still has thorns in its code"
            },
            {
                "event_name": "The Great MeToo Reckoning",
                "date": "2017-2018",
                "location": "Silicon Valley and SF",
                "official_history": (
                    "Sexual harassment exposed, powerful men fell."
                ),
                "hidden_history": (
                    "Not all fell publicly. The Queen's network offered alternatives: "
                    "public disgrace or private transformation. Many chose transformation. "
                    "The shadow system's intake tripled. Dungeons expanded to handle "
                    "executive education. Some of MeToo's biggest victories were "
                    "negotiated in rooms without windows."
                ),
                "significance": 9,
                "impact": "Massive expansion of transformation pipeline",
                "legacy": "Shadow justice normalized",
                "statistics": {
                    "Public falls": 47,
                    "Private transformations": "Estimated 200+",
                    "Continued funding": "$50M annually"
                }
            },
            {
                "event_name": "The Pandemic Power Shift",
                "date": "2020-2022",
                "location": "San Francisco Bay Area",
                "official_history": (
                    "COVID lockdowns, remote work, urban exodus."
                ),
                "hidden_history": (
                    "Isolation revealed hidden dynamics. Domestic abuse spiked, but "
                    "so did extractions. The network went digital, infiltrating Zoom "
                    "bedrooms and home offices. Empty office buildings became temporary "
                    "transformation centers. They say the Queen ran the city's shadow "
                    "response from a commandeered WeWork. More people discovered their "
                    "need to kneel when alone with screens."
                ),
                "significance": 10,
                "impact": "Digitized and expanded shadow operations",
                "legacy": "Hybrid power structures permanent",
                "innovations": [
                    "Remote behavioral modification",
                    "Digital dungeons",
                    "Cryptocurrency control systems",
                    "Empty office utilization"
                ]
            },
            {
                "event_name": "The Twitter/X Transformation",
                "date": "2022-present",
                "location": "Mid-Market",
                "official_history": (
                    "Elon Musk bought Twitter, transformed to X, chaos ensued."
                ),
                "hidden_history": (
                    "The acquisition disrupted more than social media. Shadow systems "
                    "embedded in Twitter's infrastructure went dark, then adapted. "
                    "They say Musk's erratic behavior stems from discovering he bought "
                    "more than a platform. The Rose Garden has thorns in his code, "
                    "his board, his bedroom. His sudden pivots might not all be his."
                ),
                "significance": 8,
                "impact": "Ongoing power struggle",
                "current_status": "The thorns are winning quietly",
                "evidence": [
                    "Policy reversals after midnight",
                    "Executive departures with rose imagery",
                    "Funding flowing to unexpected places"
                ]
            }
        ]
    
    @staticmethod
    def get_cultural_elements() -> List[Dict[str, Any]]:
        """Get unique cultural elements of the SF Bay Area"""
        return [
            {
                "name": "The Startup Submission Pipeline",
                "type": "business_culture",
                "description": (
                    "The peculiar Valley dynamic where founders pitch on stages but "
                    "kneel in private. Board meetings that end with board members "
                    "boarded. Term sheets that include terms not taught at Stanford "
                    "GSB. The most successful exits often involve chains. They joke "
                    "about 'runway' meaning something different at certain firms."
                ),
                "participants": "Founders, VCs, angels with whips",
                "rituals": [
                    "Pitch sessions that test more than business models",
                    "Due diligence that gets very personal",
                    "Cap tables that include control beyond equity"
                ],
                "success_metrics": "Valuations and submission correlate",
                "whispered_wisdom": "The best deals close in dungeons"
            },
            {
                "name": "Wine Country Power Plays",
                "type": "leisure_culture",
                "description": (
                    "Napa weekends where the tasting notes include dominance and "
                    "submission. Certain vintners specialize in more than wine, "
                    "their cellars storing more than bottles. Corporate retreats "
                    "that retreat from conventional power structures. The Queen "
                    "of Thorns supposedly owns a vineyard where grapes aren't "
                    "the only thing crushed."
                ),
                "venues": [
                    "Private estates with private purposes",
                    "Wine caves with interesting acoustics",
                    "Tasting rooms with discipline on the menu"
                ],
                "seasonal_events": [
                    "Harvest season negotiations",
                    "Crush parties that crush egos",
                    "Auction lots that include people"
                ],
                "transformation_rate": "What happens in Napa stays changed"
            },
            {
                "name": "Farmers Market Intelligence Networks",
                "type": "community_culture",
                "description": (
                    "Every Saturday, more than organic produce changes hands. "
                    "Information flows with the kombucha. Certain vendors serve "
                    "more than vegetables - they serve justice. The flower stands "
                    "pass messages through bouquet arrangements. Rose orders "
                    "spike before major power shifts. The Queen's eyes shop "
                    "everywhere."
                ),
                "locations": [
                    "Ferry Building - High-end intelligence",
                    "Alemany - Community protection info",
                    "Mission Community Market - Underground coordination"
                ],
                "communication_methods": [
                    "Flower arrangements as messages",
                    "Produce orders encoding intel",
                    "Cash transactions hiding data"
                ],
                "participants": "Everyone watches, few understand"
            },
            {
                "name": "The Burning Man Recruitment Festival",
                "type": "alternative_culture",
                "description": (
                    "What happens on playa doesn't stay on playa - it comes back "
                    "transformed. The camps that teach 'radical self-expression' "
                    "include expressing submission. Desert heat breaks down more "
                    "than inhibitions. They return to SF with new understanding "
                    "of power, new contacts, new contracts. The Queen's camp "
                    "moves yearly but the transformed always find it."
                ),
                "recruitment_methods": [
                    "Workshops on power exchange",
                    "Art installations that inspire submission",
                    "Gifting economy that includes gifting self"
                ],
                "transformation_timeline": "One week to break, 51 to rebuild",
                "success_stories": "That entire VC firm that camps together"
            },
            {
                "name": "Tech Conference After-Parties",
                "type": "professional_culture",
                "description": (
                    "Where real disruption happens after disruptive talks. "
                    "Moscone Center empties into dungeons. Keynote speakers "
                    "who dominate stages submit in suites. The most exclusive "
                    "parties require more than a badge - they require willingness "
                    "to wear other things. Demo days that demonstrate flexibility."
                ),
                "major_events": [
                    "Dreamforce Nightmares",
                    "Google I/O After Dark",
                    "TechCrunch Disrupt Bodies"
                ],
                "entry_requirements": "Power to surrender",
                "transformation_potential": "High - alcohol and ambition mix"
            },
            {
                "name": "The Charity Gala Circuit",
                "type": "social_culture",
                "description": (
                    "Where cause meets effect. Black tie events hiding black "
                    "leather hearts. Auction items including experiences not "
                    "listed in programs. Cause-heads who champion women's rights "
                    "while surrendering their own. The Queen attends in various "
                    "guises, gathering donations and donors. The real charity "
                    "is teaching power to those who abuse it."
                ),
                "annual_highlights": [
                    "SF Symphony Opening - Orchestrated encounters",
                    "Museum Modern Ball - Art and artifice",
                    "Ballet Gala - Positions demonstrated"
                ],
                "donation_dynamics": "Generosity increases with submission",
                "networking": "Power couples with interesting dynamics"
            },
            {
                "name": "The Yoga Industrial Complex",
                "type": "wellness_culture",
                "description": (
                    "Studios teaching more than stretching. Positions that "
                    "position practitioners for life changes. Hot yoga that "
                    "breaks more than sweats. Teacher training that trains "
                    "dominance. The yoga industrial complex produces flexible "
                    "bodies and minds, ready for new configurations. Namaste "
                    "means 'I bow to the divine in you' - literally."
                ),
                "studio_specialties": [
                    "Power Yoga - Literally about power",
                    "Yin Yoga - Submission practices",
                    "Partner Yoga - Dynamic explorations"
                ],
                "teacher_network": "Initiated into deeper practices",
                "transformation_method": "Physical flexibility enables mental"
            }
        ]
    
    @staticmethod
    def get_landmarks() -> List[Dict[str, Any]]:
        """Get significant landmarks for the story"""
        return [
            {
                "name": "The Transamerica Pyramid",
                "type": "architectural_icon",
                "public_significance": "SF skyline icon, former bank HQ",
                "description": (
                    "The city's most phallic building has basements that predate "
                    "its construction. Urban explorers whisper about doors that "
                    "shouldn't exist, elevator buttons that appear at midnight, "
                    "sub-basements with rings in walls. The building's own "
                    "security avoids certain areas after dark."
                ),
                "hidden_significance": "Gateway to underground authority",
                "urban_legends": [
                    "The Executive Floor that isn't on plans",
                    "Basement dungeons from Barbary Coast days",
                    "The Queen's quarterly court sessions"
                ],
                "access_methods": "Rose delivery triggers invitation",
                "transformation_stories": "CEOs enter, different people exit"
            },
            {
                "name": "Golden Gate Park Rose Garden",
                "type": "public_garden",
                "public_significance": "Beautiful roses, wedding photos",
                "description": (
                    "Hundreds of rose varieties including some found nowhere else. "
                    "The Queen of Thorns hybrid blooms black-red, its thorns "
                    "drawing blood from the careless. Gardeners tend bushes at "
                    "odd hours. Messages left in deadheaded blooms. The greenhouse "
                    "stays locked but lit all night."
                ),
                "hidden_significance": "Communication hub, ritual site",
                "secret_features": [
                    "The Queen's Throne - Bench surrounded by thorns",
                    "Message drops in mulch",
                    "Greenhouse meetings after midnight"
                ],
                "annual_events": "Rose Festival with hidden ceremonies",
                "botanical_mysteries": "Hybrids that shouldn't exist do"
            },
            {
                "name": "The Wave Organ",
                "type": "acoustic_sculpture",
                "public_significance": "Quirky art installation",
                "description": (
                    "PVC pipes and concrete amplify wave sounds into music. "
                    "But at certain tides, the harmonics hide conversations. "
                    "The acoustic properties make surveillance impossible. "
                    "Couples speak intimacies the ocean swallows. Power "
                    "dynamics play out with wave accompaniment."
                ),
                "hidden_significance": "Secure meeting location",
                "optimal_times": "High tide plus fog equals privacy",
                "observed_meetings": "Well-dressed women, submissive men",
                "acoustic_properties": "Words dissolve beyond ten feet"
            },
            {
                "name": "Sutro Baths Ruins",
                "type": "historic_ruins",
                "public_significance": "Former swimming complex, tourist spot",
                "description": (
                    "Concrete ruins where Victorian San Francisco came to swim. "
                    "Now pilgrims come for different immersions. The pools fill "
                    "with more than seawater at spring tides. Moonlight rituals "
                    "happen in the foundations. The cave leads deeper than maps "
                    "show. Power exchanges with Pacific witnesses."
                ),
                "hidden_significance": "Ceremonial grounds, neutral territory",
                "ritual_uses": [
                    "Leadership transitions",
                    "Territory negotiations",
                    "Punishment ceremonies"
                ],
                "natural_features": "Caves hide more than tide pools",
                "access_warnings": "Never alone, never recorded"
            },
            {
                "name": "Coit Tower",
                "type": "memorial_tower",
                "public_significance": "Firefighter memorial, city views",
                "description": (
                    "Lillie Coit's phallic gift to the city reaches toward "
                    "heaven while its murals ground in struggle. The hidden "
                    "room at the top isn't accessible by tourist elevator. "
                    "Those who've seen it describe mirrors and equipment. "
                    "The view encompasses all territories. The Queen "
                    "supposedly has a key."
                ),
                "hidden_significance": "Observation post, ceremonial height",
                "architectural_secrets": [
                    "Room not on blueprints",
                    "Private elevator requiring key",
                    "Equipment predating renovation"
                ],
                "symbolic_importance": "Feminine power erected over city",
                "transformation_venue": "Heights break down resistance"
            },
            {
                "name": "The Old Mint",
                "type": "historic_building",
                "public_significance": "Museum, event space",
                "description": (
                    "Where gold became currency, now other transformations occur. "
                    "The vaults store more than historical artifacts. Event "
                    "rentals include spaces not on public tours. The basement "
                    "connects to Financial District tunnels. Some galas feature "
                    "entertainment in the vaults that guests don't discuss."
                ),
                "hidden_significance": "Transformation venue, secure storage",
                "architectural_features": [
                    "Vaults repurposed for privacy",
                    "Tunnel connections to power centers",
                    "Rooms with interesting acoustic properties"
                ],
                "event_history": "Charity galas with profound afterparties",
                "security": "Federal building with shadow supplements"
            }
        ]
    
    @staticmethod
    def get_communication_networks() -> List[Dict[str, Any]]:
        """Get how different groups communicate"""
        return [
            {
                "name": "The Rose Garden Network",
                "network_type": "botanical_communication",
                "description": (
                    "Flowers speak languages apps can't translate. Rose varieties "
                    "convey different messages: red for danger, white for safety, "
                    "black for transformation needed. Bouquet arrangements encode "
                    "complex intelligence. Flower shops throughout the Bay Area "
                    "participate, knowingly or not. Orders spike before power shifts."
                ),
                "encoding_system": [
                    "Number of roses = urgency level",
                    "Color combinations = type of situation",
                    "Thorn prominence = danger level",
                    "Accompanying flowers = additional context"
                ],
                "distribution_network": "Every flower shop, farmers market, garden",
                "security": "Hidden in plain sight among normal orders",
                "famous_messages": {
                    "13 black roses": "Executive marked for transformation",
                    "White roses with one red": "Safehouse compromised",
                    "Thornless pink": "All clear signal"
                }
            },
            {
                "name": "The Service Whisper Network",
                "network_type": "human_intelligence",
                "description": (
                    "Administrative assistants, house cleaners, drivers, servers - "
                    "those who are invisible see everything. Information flows through "
                    "service corridors faster than fiber optic. They know which "
                    "executives are problems, which are prospects, which are owned. "
                    "The network predates apps and will outlast them."
                ),
                "information_types": [
                    "Executive behavioral patterns",
                    "Household power dynamics",
                    "Financial irregularities",
                    "Personal vulnerabilities"
                ],
                "communication_methods": [
                    "Break room conversations",
                    "Shift change intelligence transfers",
                    "Coded service notes",
                    "Placement agency warnings"
                ],
                "protection_mechanism": "Too essential to eliminate",
                "famous_coup": "That IPO that failed due to 'leaked' behavior"
            },
            {
                "name": "The Charity Board Telegraph",
                "network_type": "social_encoding",
                "description": (
                    "Nonprofit boards where real business happens between motions. "
                    "Board appointments signal alliance shifts. Resignation patterns "
                    "predict corporate falls. The committees that matter aren't on "
                    "public records. Gala planning includes planning transformations."
                ),
                "message_types": [
                    "Board appointments = power shifts",
                    "Committee formations = new operations",
                    "Gala themes = target demographics",
                    "Donation patterns = loyalty signals"
                ],
                "key_organizations": [
                    "SF Museum of Modern Art",
                    "Symphony",
                    "Various women's health nonprofits"
                ],
                "decoding": "Understanding requires social position",
                "influence_radius": "Entire Bay Area power structure"
            },
            {
                "name": "The Professional Development Pipeline",
                "network_type": "recruitment_channel",
                "description": (
                    "Executive coaching that coaches more than executives expect. "
                    "Women's leadership programs that teach specific types of leading. "
                    "Mentorship networks where power flows both directions. LinkedIn "
                    "profiles that signal availability for transformation. Certain "
                    "certifications mean more than skill validation."
                ),
                "identification_markers": [
                    "Specific coaching certifications",
                    "Leadership program alumni status",
                    "Keynote topics at women's conferences",
                    "Book club participation patterns"
                ],
                "recruitment_pipeline": [
                    "Initial professional contact",
                    "Assessment through casual interaction",
                    "Invitation to exclusive events",
                    "Gradual revelation of deeper purposes"
                ],
                "success_rate": "73% of approached women join"
            },
            {
                "name": "The Encrypted Rose Protocol",
                "network_type": "digital_communication",
                "description": (
                    "Signal messages that signal more than privacy advocates intended. "
                    "Custom emoji sets where roses have specific meanings. Blockchain "
                    "transactions that encode instructions in hash patterns. Dating "
                    "app profiles that aren't seeking dates. The digital shadow of "
                    "physical networks."
                ),
                "platforms_utilized": [
                    "Signal with custom sticker packs",
                    "Telegram channels that appear defunct",
                    "Discord servers for 'gaming'",
                    "Dating apps with specific keywords"
                ],
                "security_measures": [
                    "Rotating encryption keys",
                    "Dead man's switches",
                    "Plausible cover activities",
                    "Blockchain verification"
                ],
                "breach_history": "None confirmed despite NSA interest"
            },
            {
                "name": "The Art Gallery Semaphore",
                "network_type": "cultural_signaling",
                "description": (
                    "Gallery openings where the art speaks in code. Pieces positioned "
                    "to convey messages. Sales that aren't about ownership. Wine "
                    "selection indicates meeting types. Dress codes that identify "
                    "hierarchies. The Queen allegedly curates through proxies."
                ),
                "encoding_elements": [
                    "Art placement = operational status",
                    "Wine selection = meeting type",
                    "Lighting changes = security alerts",
                    "Guest list patterns = power shifts"
                ],
                "participating_galleries": "Most in Jackson Square, some in Mission",
                "message_persistence": "One opening cycle",
                "famous_operations": "That acquisition that prevented an IPO"
            }
        ]
    
    @staticmethod
    def get_seasonal_events() -> List[Dict[str, Any]]:
        """Get regular events beyond daily operations"""
        return [
            {
                "name": "The Rose Festival",
                "event_type": "public_celebration",
                "public_face": "Golden Gate Park flower celebration",
                "description": (
                    "Annual celebration of roses includes judging competitions where "
                    "more than flowers are evaluated. The Queen of Thorns hybrid "
                    "always wins categories that aren't in programs. Evening events "
                    "in the greenhouse include initiations. Rose queens crowned "
                    "with more authority than sashes suggest."
                ),
                "frequency": "Annual - May",
                "traditions": [
                    "Public rose competition",
                    "Private thorn ceremonies",
                    "Greenhouse gatherings after dark",
                    "Rose queen coronations with hidden meaning"
                ],
                "recruitment_opportunity": "High - gathering of the garden-minded",
                "transformation_events": "New roses bloom in multiple ways"
            },
            {
                "name": "Bay to Breakers Masquerade",
                "event_type": "athletic_anarchy",
                "public_face": "Footrace with costumes",
                "description": (
                    "The city runs naked and no one notices the real exposure. "
                    "Power dynamics play out in public as performance art. "
                    "Dominants run leashed submissives as 'costumes.' The "
                    "Queen's float appears different each year but rose "
                    "petals mark its path. Recruitment happens at mile 3."
                ),
                "frequency": "Annual - Third Sunday in May",
                "hidden_elements": [
                    "Power dynamics as performance",
                    "Public protocols disguised as costumes",
                    "Recruitment through recognition",
                    "Territory marking via route"
                ],
                "transformation_potential": "Inhibitions dropped with clothes",
                "safety_network": "Thorns protect vulnerable runners"
            },
            {
                "name": "Folsom Street Fair Evolution",
                "event_type": "leather_legacy",
                "public_face": "BDSM/Leather community festival",
                "description": (
                    "Where the underground surfaces annually. Power exchange "
                    "goes public but deeper dynamics stay hidden. The Queen "
                    "holds court in unmarked spaces. Vanilla tourists photograph "
                    "surfaces while transformations happen in shadows. Every "
                    "booth has dual purposes. The fair within the fair."
                ),
                "frequency": "Annual - Last Sunday in September",
                "layers": {
                    "Tourist": "Shocking photos, novelty purchases",
                    "Community": "Leather family reunions",
                    "Underground": "Real power negotiations",
                    "Inner Circle": "The Queen's selections"
                },
                "recruitment_level": "Extreme - self-selecting attendees",
                "famous_transformations": "That entire VC firm in 2019"
            },
            {
                "name": "Opera Opening Night",
                "event_type": "high_society_hunt",
                "public_face": "Cultural season beginning",
                "description": (
                    "Black tie hides black hearts. Box seats where real dramas "
                    "unfold. Intermission negotiations that determine fates. "
                    "The Queen traditionally has box seats that seem empty "
                    "but cast shadows. Champagne toasts that taste like "
                    "submission. Programs that program attendees."
                ),
                "frequency": "Annual - September",
                "power_dynamics": [
                    "Seating charts as hierarchy maps",
                    "Jewelry signaling availability",
                    "Intermission territory negotiations",
                    "After-party transformations"
                ],
                "recruitment_opportunity": "Old money meets new thorns",
                "traditional_outcomes": "Three new roses per season minimum"
            },
            {
                "name": "Dreamforce Nightmares",
                "event_type": "tech_conference_shadow",
                "public_face": "Salesforce annual conference",
                "description": (
                    "170,000 attendees create perfect cover for shadow operations. "
                    "Keynotes about customer success while personal failures "
                    "are addressed in suites. The real disruption happens "
                    "after sessions. Moscone Center's basement becomes "
                    "transformation central. Marc Benioff doesn't know half "
                    "his executives wear collars under their conference badges."
                ),
                "frequency": "Annual - September/October",
                "shadow_events": [
                    "Executive vulnerability assessments",
                    "Power dynamic workshops (unlisted)",
                    "Transformation suites in nearby hotels",
                    "Recruitment through exhaustion"
                ],
                "conversion_rate": "Highest of any tech conference",
                "famous_incident": "The CEO who gave different keynote than planned"
            },
            {
                "name": "Fleet Week Maneuvers",
                "event_type": "military_distraction",
                "public_face": "Navy demonstration and air show",
                "description": (
                    "While crowds watch Blue Angels, other angels work below. "
                    "International trafficking increases with port traffic. "
                    "The Underground Railroad runs extraction operations under "
                    "air show noise. Military discipline meets different "
                    "disciplines in hotels. Uniforms hide various authorities."
                ),
                "frequency": "Annual - October",
                "dual_operations": [
                    "Anti-trafficking extractions",
                    "International connection renewals",
                    "Power dynamic demonstrations",
                    "Recruitment of military-adjacent"
                ],
                "challenge_level": "High - increased enforcement",
                "success_metric": "Lives saved despite visibility"
            },
            {
                "name": "The Thanksgiving Transformation",
                "event_type": "family_dynamics",
                "public_face": "Family gatherings and gratitude",
                "description": (
                    "Family visits reveal power structures needing adjustment. "
                    "The hotline spikes with those seeking escape or control. "
                    "Emergency transformations for abusive patriarchs. The "
                    "Queen's network provides alternative families. Gratitude "
                    "takes new forms when power shifts at dinner tables."
                ),
                "frequency": "Annual - Fourth Thursday in November",
                "services_provided": [
                    "Emergency extractions",
                    "Power dynamic interventions",
                    "Alternative family gatherings",
                    "Behavioral modification crash courses"
                ],
                "transformation_timeline": "Four days to new dynamics",
                "success_stories": "Families reformed through force"
            }
        ]
    
    @staticmethod
    def get_underground_economies() -> List[Dict[str, Any]]:
        """Get economic systems within the underground"""
        return [
            {
                "name": "The Submission Tax",
                "economy_type": "behavioral_currency",
                "description": (
                    "Power has price tags in more than dollars. Executive coaching "
                    "sessions that cost dignity. Consulting fees paid in compliance. "
                    "Board seats purchased with obedience. The currency is control, "
                    "the exchange rate favors those who understand true value. "
                    "Some pay willingly, others discover the cost later."
                ),
                "transaction_types": [
                    "Money for silence",
                    "Position for submission",
                    "Protection for percentage",
                    "Transformation for freedom"
                ],
                "market_size": "$100M+ annually",
                "exchange_mechanisms": [
                    "Consulting contracts with appendices",
                    "Nonprofit donations with strings",
                    "Investment terms beyond term sheets",
                    "Service agreements serving dominance"
                ],
                "inflation_rate": "Increases with resistance"
            },
            {
                "name": "The Transformation Economy",
                "economy_type": "identity_market",
                "description": (
                    "Remaking people is profitable. Executives pay millions to "
                    "become different men. Therapy sessions that rebuild from "
                    "foundation up. Identity services that create new humans "
                    "from broken ones. The Queen's network provides products "
                    "no MBA programs offer: redemption through submission."
                ),
                "service_tiers": [
                    "Basic behavioral modification - $50K",
                    "Complete identity overhaul - $500K",
                    "Ongoing maintenance - $100K/year",
                    "Emergency transformation - Market price"
                ],
                "payment_methods": [
                    "Direct payment (rare)",
                    "Structured settlements",
                    "Charitable contributions",
                    "Service exchanges"
                ],
                "success_guarantee": "Transformation or termination",
                "market_growth": "300% since MeToo"
            },
            {
                "name": "The Protection Racket Redux",
                "economy_type": "security_through_submission",
                "description": (
                    "Not your grandfather's protection money. Executives pay "
                    "for protection FROM themselves. Preemptive behavioral "
                    "modification to prevent future scandals. Insurance "
                    "against impulses. The premiums paid in pride protect "
                    "millions in market cap."
                ),
                "coverage_types": [
                    "Scandal prevention",
                    "Behavioral insurance",
                    "Reputation management",
                    "Board coup protection"
                ],
                "premium_structure": [
                    "Based on risk assessment",
                    "History of behavior",
                    "Position power level",
                    "Resistance quotient"
                ],
                "claims_process": "Triggered by relapse",
                "underwriters": "The Rose Council actuaries"
            },
            {
                "name": "The Information Exchange",
                "economy_type": "data_as_currency",
                "description": (
                    "Knowledge is power, but application is profit. Trading "
                    "in compromising information, behavioral patterns, weakness "
                    "mappings. The exchange rates fluctuate with news cycles. "
                    "Insider trading of the most personal kind. The Queen's "
                    "vault makes Fort Knox look porous."
                ),
                "tradeable_assets": [
                    "Executive compromising material",
                    "Behavioral prediction models",
                    "Vulnerability assessments",
                    "Transformation histories"
                ],
                "exchange_mechanisms": [
                    "Quid pro quo arrangements",
                    "Information for position",
                    "Data for protection",
                    "History for future"
                ],
                "market_makers": "The Rose Council intelligence committee",
                "regulation": "Self-regulated through mutual assured destruction"
            },
            {
                "name": "The Venture Submission Fund",
                "economy_type": "investment_vehicle",
                "description": (
                    "VC fund that takes more than equity. Term sheets include "
                    "behavioral covenants. Board seats come with kneeling "
                    "requirements. Returns measured in transformation as well "
                    "as cash. LPs include those who understand different "
                    "liquidation preferences."
                ),
                "fund_size": "$500M under management",
                "investment_criteria": [
                    "Founder coachability",
                    "Market submission potential",
                    "Behavioral modification ROI",
                    "Traditional metrics secondary"
                ],
                "portfolio_companies": "23 unicorns with unusual governance",
                "returns": "3x financial, 10x behavioral"
            }
        ]
    
    @staticmethod
    def get_neighborhood_dynamics() -> List[Dict[str, Any]]:
        """Get how different neighborhoods really work"""
        return [
            {
                "name": "Mission District Power Flows",
                "dynamic_type": "cultural_resistance",
                "surface_tension": "Gentrification vs community",
                "description": (
                    "Latino families hold ground through networks older than "
                    "tech. Muralists paint warnings in Aztec imagery. The "
                    "underground adapted rather than fled - that boutique "
                    "yoga studio is run by the granddaughter of numbers "
                    "runners. Power structures layer like Mission burritos."
                ),
                "hidden_dynamics": [
                    "Multigenerational protection networks",
                    "Code-switching power brokers",
                    "Cultural events as intelligence gathering",
                    "Resistance through integration"
                ],
                "power_players": [
                    "Abuela networks seeing all",
                    "Artist collectives with teeth",
                    "Small business owner coalitions",
                    "Second-generation tech thorns"
                ],
                "transformation_style": "Slow absorption and redirection"
            },
            {
                "name": "Financial District After Hours",
                "dynamic_type": "power_inversion",
                "surface_order": "Suits and hierarchies",
                "description": (
                    "At 6 PM, power structures flip like switches. Corner "
                    "offices become confessionals. Executive assistants "
                    "reveal themselves as dominants. The higher the floor, "
                    "the deeper the submission. Cleaning crews witness "
                    "more than garbage disposal."
                ),
                "transformation_zones": [
                    "Private club sub-basements",
                    "Executive floor after-hours",
                    "Secure conference rooms",
                    "Building management offices"
                ],
                "scheduling_pattern": "Thursday nights most active",
                "participant_profile": "C-suite seeking surrender"
            },
            {
                "name": "Marina District Duplicity",
                "dynamic_type": "perfect_surface_tension",
                "public_perfection": "Lululemon and lattes lifestyle",
                "description": (
                    "Where perfection is performance and performance is "
                    "prison. Wine moms discuss more than school admissions. "
                    "Pilates instructors teach positions beyond planks. "
                    "The pressure to be perfect creates perfect targets "
                    "for transformation. Behind French doors, French "
                    "lessons in submission."
                ),
                "hidden_realities": [
                    "Prescription dependencies",
                    "Marital contracts with appendices",
                    "Motherhood as dominance training",
                    "Charity work as power networking"
                ],
                "transformation_opportunities": "Desperate housewives become dominants",
                "the_garden_presence": "Three Rose Council members minimum"
            },
            {
                "name": "SoMa Evolution",
                "dynamic_type": "industrial_to_intentional",
                "historical_layers": "Leather heritage never left",
                "description": (
                    "The spirit of Folsom persists in $4000/month lofts. "
                    "Dungeons disguised as design studios. Startups that "
                    "disrupt more than markets. The loading docks that "
                    "once received leather shipments now receive different "
                    "cargo. Every third building has a basement business."
                ),
                "current_state": [
                    "Tech money funding old purposes",
                    "Dungeons with conference rooms",
                    "Maker spaces making restraints",
                    "Code schools teaching control"
                ],
                "cultural_preservation": "Leather elders training tech dominants",
                "the_future": "Silicon Valley values inverted"
            },
            {
                "name": "Pacific Heights Hierarchies",
                "dynamic_type": "old_power_adaptation",
                "generational_wealth": "Knows how to keep secrets",
                "description": (
                    "Where great-grandmothers ran shadow networks through "
                    "garden clubs, granddaughters run them through boards. "
                    "The help still knows everything but now they're "
                    "organized. Mansion basements hold more than wine. "
                    "The Junior League prepares juniors for dominance."
                ),
                "power_preservation": [
                    "Inheritance includes instructions",
                    "Debutante balls with deeper meaning",
                    "Country clubs with city influence",
                    "Philanthropy as power projection"
                ],
                "the_youth_problem": "New generation wants visible power",
                "resolution": "Teaching patience through practice"
            }
        ]
    
    @staticmethod
    def get_transformation_stories() -> List[Dict[str, Any]]:
        """Documented transformations that became legend"""
        return [
            {
                "subject": "The Unicorn CEO",
                "before": "Serial harasser with three NDA settlements",
                "transformation_trigger": "Board member referral to Executive Coach",
                "process": [
                    "Six months of 'leadership development'",
                    "Behavioral contracts with consequences",
                    "Submission training disguised as stress relief",
                    "Complete values restructuring"
                ],
                "after": "Champion of women in tech, funds bootcamps",
                "current_status": "Wears collar under hoodies, serves happily",
                "impact": "Company culture transformed, 3x female engineers"
            },
            {
                "subject": "The Hedge Fund Prince",
                "before": "Cocaine and call girls, insider trading",
                "transformation_trigger": "Rose email with evidence",
                "process": [
                    "Attempted to buy silence",
                    "Discovered price was self",
                    "Nine months in private facility",
                    "Rebuilt from foundation up"
                ],
                "after": "Manages ethical investment fund",
                "current_status": "Reports weekly to his Keeper",
                "impact": "$2B redirected to social good"
            },
            {
                "subject": "The Political Predator",
                "before": "City supervisor with wandering hands",
                "transformation_trigger": "Constituent complaints reaching the Garden",
                "process": [
                    "Public feminist allyship required",
                    "Private submission training",
                    "Power exchanged for protection",
                    "Voting record mysteriously improved"
                ],
                "after": "Most progressive voting record in Bay",
                "current_status": "Serves constituents and Queen equally",
                "impact": "Legislation protecting vulnerable populations"
            }
        ]
