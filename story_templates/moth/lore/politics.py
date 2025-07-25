# story_templates/moth/lore/politics.py
"""
Political issues, conflicts, and power dynamics for SF Bay Area
"""

from typing import Dict, Any, List

class SFPoliticsLore:
    """Political and domestic issues for SF Bay Area"""
    
    @staticmethod
    def get_domestic_issues() -> List[Dict[str, Any]]:
        """Get local political conflicts"""
        return [
            {
                "name": "The Housing Affordability Crisis",
                "issue_type": "economic_social",
                "description": (
                    "Median home price over $1.5 million, teachers commuting 3 hours, "
                    "tent cities under freeways. The crisis deepens monthly. But certain "
                    "development projects mysteriously fail. Environmental reviews find "
                    "endangered butterflies. Investors suddenly pull funding. Key opponents "
                    "have unexpected changes of heart. Someone protects certain communities."
                ),
                "severity": 10,
                "status": "chronic_crisis",
                "public_debate": {
                    "YIMBY_position": "Build more housing everywhere",
                    "NIMBY_position": "Preserve neighborhood character",
                    "Tenant_position": "Strengthen rent control",
                    "Developer_position": "Reduce regulations"
                },
                "hidden_dynamics": {
                    "Thorn_Garden_position": "Selective intervention",
                    "Protected_communities": "Artists, minorities, essential workers",
                    "Methods": [
                        "Environmental lawsuits appear from nowhere",
                        "Investors receive troubling information",
                        "Zoning officials have personal revelations",
                        "Community organizers get unexpected funding"
                    ],
                    "The_Queen_says": "A city without soul houses no one"
                },
                "key_players": {
                    "Mayor_Patricia_Chen": "Publicly frustrated, privately informed",
                    "Supervisor_Districts": "Some influenced, some oblivious",
                    "Developer_Magnus_Thornwood": "Learning respect the hard way",
                    "Housing_advocates": "Mysteriously well-funded lately"
                },
                "recent_events": [
                    "Luxury tower in Mission cancelled after developer scandal",
                    "Affordable housing in Bayview fast-tracked mysteriously",
                    "Three slumlords fled city after 'personal issues'",
                    "Teacher housing initiative funded by anonymous donor"
                ]
            },
            {
                "name": "Tech Industry Regulation Wars",
                "issue_type": "regulatory_economic",
                "description": (
                    "City attempts to regulate Uber, Airbnb, delivery apps, scooters. "
                    "Tech companies threaten to leave, taking jobs and tax revenue. "
                    "Public hearings get heated. But in private meetings, certain "
                    "executives become surprisingly cooperative. Others find their "
                    "personal lives complicating their corporate positions."
                ),
                "severity": 8,
                "status": "ongoing_negotiation",
                "public_positions": {
                    "Tech_companies": "Innovation requires freedom",
                    "City_government": "Public safety and worker rights",
                    "Unions": "Protect workers from gig exploitation",
                    "Residents": "Split between convenience and concerns"
                },
                "shadow_negotiations": {
                    "Real_meetings": "Not in City Hall but private clubs",
                    "Leverage_used": [
                        "Executive personal scandals",
                        "Board member conflicts",
                        "Investor nervousness",
                        "Family pressures"
                    ],
                    "Mediators": "Women who know both worlds",
                    "Outcomes": "Compromises that surprise analysts"
                },
                "specific_battles": [
                    {
                        "issue": "Uber driver classification",
                        "public_fight": "Prop 22 and lawsuits",
                        "private_resolution": "CEO's midnight conversation",
                        "result": "Better local deal than expected"
                    },
                    {
                        "issue": "Airbnb housing impact",
                        "public_fight": "Registration requirements",
                        "private_pressure": "Host network infiltrated",
                        "result": "Certain neighborhoods protected"
                    }
                ],
                "Queen_network_position": "Technology serves humanity, not vice versa"
            },
            {
                "name": "The Homeless Crisis Response",
                "issue_type": "humanitarian_political",
                "description": (
                    "40,000+ unhoused residents, sweeps vs services debate raging. "
                    "Navigation Centers vs shelter beds. Public divided between "
                    "compassion and frustration. But certain camps never get swept. "
                    "Some homeless find resources appearing. Underground networks "
                    "protect the most vulnerable."
                ),
                "severity": 9,
                "status": "critical",
                "visible_debate": {
                    "Progressive_position": "Housing first, services not sweeps",
                    "Moderate_position": "Balance compassion with public safety",
                    "Business_position": "Clear the streets for commerce",
                    "Advocate_position": "Human rights before property rights"
                },
                "invisible_interventions": [
                    "Vulnerable women extracted from camps",
                    "Predators in camps mysteriously vanish",
                    "Resources appear through 'faith groups'",
                    "Certain areas designated as protected",
                    "Underground railroad for trafficking victims"
                ],
                "network_operations": {
                    "Safe_zones": "Marked with subtle symbols",
                    "Extraction_teams": "Social workers with backup",
                    "Resource_drops": "Anonymous but regular",
                    "Intelligence": "Who's dangerous, who needs help"
                },
                "political_complexity": "Can't publicly admit shadow operations"
            },
            {
                "name": "Police Reform and Public Safety",
                "issue_type": "law_enforcement",
                "description": (
                    "Post-2020 reform efforts stalled. Progressive DA faces recall. "
                    "Crime stats weaponized by all sides. SFPD budget battles annual. "
                    "But certain crimes never make statistics. Alternative justice "
                    "systems operate in shadows. Some officers know to look away."
                ),
                "severity": 8,
                "status": "volatile",
                "public_factions": {
                    "Reformers": "Defund and replace with services",
                    "Moderates": "Reform but maintain presence",
                    "Law_and_order": "More cops, tougher enforcement",
                    "Abolitionists": "Community solutions only"
                },
                "shadow_justice": {
                    "Crimes_handled_privately": [
                        "Domestic violence with connected abusers",
                        "Sexual assault with powerful perpetrators",
                        "Trafficking operations",
                        "Predatory behavior"
                    ],
                    "Methods": [
                        "Perpetrators vanish or confess",
                        "Evidence appears perfectly packaged",
                        "Witnesses protected invisibly",
                        "Justice without courts"
                    ],
                    "Police_awareness": "Some cops grateful, others frustrated"
                },
                "DA_recall_complexity": {
                    "Public_reason": "Crime and prosecution rates",
                    "Hidden_factor": "DA knows about alternative systems",
                    "Network_position": "Neutral - both candidates compromised"
                }
            },
            {
                "name": "Climate Action vs Economic Reality",
                "issue_type": "environmental_economic",
                "description": (
                    "Net zero goals vs tech growth. Sea level rise threatens billions "
                    "in real estate. Wildfires smoke the city annually. But green "
                    "initiatives pass or fail based on hidden factors. Some developers "
                    "suddenly embrace sustainability. Others find their projects "
                    "haunted by environmental violations."
                ),
                "severity": 7,
                "status": "long_term_crisis",
                "public_tensions": [
                    "Green New Deal vs job concerns",
                    "Transit expansion vs car culture",
                    "Density for climate vs neighborhood preservation",
                    "Corporate pledges vs actual action"
                ],
                "Garden_influence": {
                    "Blessed_projects": "True sustainability supported",
                    "Cursed_developments": "Greenwashing exposed",
                    "Executive_conversions": "CEOs discover earth matters",
                    "Funding_flows": "Anonymous green donations"
                },
                "specific_interventions": [
                    "Offshore oil executive's change of heart",
                    "Green building standards mysteriously strengthened",
                    "Climate deniers find careers complicated",
                    "Sustainable businesses get unexpected help"
                ]
            },
            {
                "name": "The Port Authority Scandal",
                "issue_type": "corruption_investigation",
                "description": (
                    "Federal investigation into Port of Oakland smuggling. Trafficking "
                    "rumors persist. Commissioner Huang's reforms seem cosmetic. But "
                    "certain shipments never arrive. Key witnesses develop amnesia. "
                    "The investigation stalls in interesting ways."
                ),
                "severity": 9,
                "status": "active_investigation",
                "public_knowledge": {
                    "Official_line": "Isolated corruption being addressed",
                    "Media_coverage": "Sporadic and confused",
                    "Federal_position": "Ongoing investigation",
                    "Port_response": "Full cooperation claimed"
                },
                "actual_situation": {
                    "Trafficking_routes": "Still active but disrupted",
                    "Garden_operations": [
                        "Intelligence gathering on shipments",
                        "Extraction teams at ready",
                        "Document trails created/destroyed",
                        "Witnesses protected or pressured"
                    ],
                    "Commissioner_Huang": "Playing three sides minimum",
                    "The_real_question": "How deep does corruption go?"
                },
                "network_dilemma": "Expose corruption or protect operations?",
                "recent_developments": [
                    "Three containers 'lost' last month",
                    "Dock worker promoted after 'lottery win'",
                    "Federal agent transferred suddenly",
                    "Anonymous tips keep appearing"
                ]
            }
        ]
    
    @staticmethod
    def get_political_figures() -> List[Dict[str, Any]]:
        """Get key political players and their hidden connections"""
        return [
            {
                "name": "Mayor Patricia Chen",
                "position": "Mayor of San Francisco",
                "public_persona": {
                    "image": "Progressive pragmatist",
                    "platform": "Housing, homelessness, climate action",
                    "style": "Diplomatic but firm",
                    "approval": "48% - polarizing figure"
                },
                "hidden_connections": {
                    "Garden_awareness": "Knows something exists",
                    "Personal_history": "College roommate vanished into network",
                    "Pressure_points": "Daughter at Stanford, asking questions",
                    "Usefulness": "Can be guided, not controlled"
                },
                "recent_actions": [
                    "Killed development that would displace artists",
                    "Appointed sympathetic police commissioner",
                    "Attends certain charity events regularly",
                    "Late night meetings off the books"
                ],
                "network_assessment": "Potential ally, requires careful handling"
            },
            {
                "name": "Supervisor Jessica Martinez",
                "position": "District 9 Supervisor (Mission)",
                "public_persona": {
                    "image": "Firebrand activist turned politician",
                    "platform": "Tenant rights, immigrant protection",
                    "base": "Latino community, progressive activists",
                    "reputation": "Uncompromising fighter"
                },
                "hidden_truth": {
                    "Network_role": "Unknowing asset",
                    "Influences": "Chief of staff is Garden member",
                    "Protection": "Threats against her handled quietly",
                    "Potential": "Being evaluated for recruitment"
                },
                "leverage_points": [
                    "Genuinely cares about community",
                    "History of personal trauma",
                    "Responds to displays of power",
                    "Ambitious beyond current role"
                ]
            },
            {
                "name": "DA Chelsea Walsh",
                "position": "District Attorney",
                "public_persona": {
                    "image": "Progressive prosecutor",
                    "policies": "Restorative justice, reduced incarceration",
                    "controversy": "Facing recall election",
                    "support": "Split between reformers and critics"
                },
                "complicated_reality": {
                    "Knowledge": "Aware of alternative justice systems",
                    "Cooperation": "Selectively doesn't prosecute",
                    "Pressure": "From both network and opponents",
                    "Dilemma": "Can't explain certain decisions"
                },
                "specific_cases": [
                    "Dropped charges against executive - had reasons",
                    "Aggressive prosecution of trafficker - had help",
                    "Witness protection beyond normal means",
                    "Evidence appearing too perfectly"
                ],
                "future": "Recall might not matter if network decides"
            },
            {
                "name": "Chief of Police William O'Brien",
                "position": "SFPD Chief",
                "public_persona": {
                    "image": "Reform-minded cop",
                    "background": "Came from Seattle with promises",
                    "challenges": "Union resistance, political pressure",
                    "style": "Data-driven, community-oriented"
                },
                "street_reality": {
                    "Awareness": "Knows about shadow justice",
                    "Position": "Pragmatic acceptance",
                    "Calculation": "Better than gang warfare",
                    "Quiet_orders": "Certain areas left alone"
                },
                "network_interaction": [
                    "Monthly off-books meeting with unknown woman",
                    "Certain cases routed to specific detectives",
                    "Resource allocation follows patterns",
                    "Personnel transfers protect operations"
                ]
            },
            {
                "name": "Judge Catherine Rosewood",
                "position": "Superior Court Judge",
                "public_record": {
                    "reputation": "Fair but firm",
                    "specialties": "Family law, domestic violence",
                    "rulings": "Consistently protective of victims",
                    "courtroom": "No nonsense tolerance"
                },
                "deeper_truth": {
                    "Network_position": "Rose Council member",
                    "Special_services": [
                        "Emergency restraining orders",
                        "Custody arrangements that protect",
                        "Sentencing that sends messages",
                        "Case assignments when needed"
                    ],
                    "Cover": "Just a tough feminist judge",
                    "Reality": "Enforces Garden law through bench"
                }
            },
            {
                "name": "Congressman Derek Thompson",
                "position": "U.S. Representative, SF District",
                "public_image": {
                    "platform": "Tech regulation, climate action",
                    "style": "Young, ambitious, presidential?",
                    "funding": "Tech and traditional donors",
                    "record": "Progressive with pragmatic votes"
                },
                "hidden_complications": {
                    "Scandal_brewing": "Intern situation handled quietly",
                    "Debt": "Owes continued silence to Garden",
                    "Behavior": "Modified after intervention",
                    "Future": "Controlled asset or exposed example"
                },
                "usefulness": "Federal influence when needed"
            }
        ]
    
    @staticmethod
    def get_political_events() -> List[Dict[str, Any]]:
        """Get ongoing political events and campaigns"""
        return [
            {
                "event_name": "DA Recall Election",
                "event_type": "special_election",
                "timeline": "Signature gathering through November",
                "public_narrative": {
                    "Pro_recall": "Crime rising, prosecution failing",
                    "Anti_recall": "Reform takes time, stay the course",
                    "Media_coverage": "Daily drama and statistics",
                    "Polling": "Too close to call"
                },
                "hidden_dynamics": {
                    "Garden_position": "Neutral with contingencies",
                    "Both_candidates": "Have vulnerabilities",
                    "Real_question": "Who can be influenced better?",
                    "Operations": "Continue regardless of outcome"
                },
                "campaign_irregularities": [
                    "Major donor suddenly withdrew",
                    "Opposition researcher quit abruptly",
                    "Key endorsement flipped overnight",
                    "Campaign manager replaced quietly"
                ],
                "potential_outcomes": {
                    "Walsh_survives": "Deeper cooperation expected",
                    "Walsh_recalled": "Replacement gets education",
                    "Either_way": "Shadow justice continues"
                }
            },
            {
                "event_name": "Mayoral Race 2027 Positioning",
                "event_type": "future_campaign",
                "early_moves": "Already beginning",
                "potential_candidates": [
                    {
                        "name": "Supervisor Martinez",
                        "strengths": "Progressive base, Latino support",
                        "weaknesses": "Limited citywide appeal",
                        "Garden_view": "Possible but needs development"
                    },
                    {
                        "name": "Tech Executive Victoria Chen",
                        "strengths": "Money, moderate appeal",
                        "weaknesses": "No political experience",
                        "Garden_view": "Already an asset"
                    },
                    {
                        "name": "Former Mayor's Chief of Staff",
                        "strengths": "Experience, connections",
                        "weaknesses": "Boring candidate",
                        "Garden_view": "Controllable but uninspiring"
                    }
                ],
                "network_strategy": "Cultivate multiple options"
            },
            {
                "event_name": "Housing Bond Measure",
                "event_type": "ballot_initiative",
                "amount": "$2 billion",
                "public_debate": [
                    "Affordable housing funding",
                    "Property tax implications",
                    "Developer giveaways?",
                    "Union support secured"
                ],
                "behind_scenes": {
                    "Language_tweaked": "Protects certain communities",
                    "Opposition_funding": "Suddenly dried up",
                    "Key_endorsements": "Obtained through pressure",
                    "Polling_shifts": "After targeted campaigns"
                },
                "expected_outcome": "Passage with Garden provisions intact"
            },
            {
                "event_name": "Tech Tax Initiative",
                "event_type": "corporate_taxation",
                "proposal": "Tax on companies above $50M revenue",
                "public_positions": {
                    "Tech_industry": "Job killer, will leave SF",
                    "Progressives": "Fair share for public services",
                    "Moderates": "Worried about economic impact",
                    "Unions": "Enthusiastic support"
                },
                "shadow_negotiations": {
                    "CEOs_approached": "Privately, individually",
                    "Pressure_applied": "Personal and professional",
                    "Compromises_suggested": "Tax for protection",
                    "Result": "Less opposition than expected"
                },
                "Garden_interest": "Revenue for influenced programs"
            }
        ]
    
    @staticmethod
    def get_media_landscape() -> List[Dict[str, Any]]:
        """How information flows and is controlled"""
        return [
            {
                "outlet": "San Francisco Chronicle",
                "outlet_type": "newspaper_of_record",
                "ownership": "Hearst Corporation",
                "public_role": {
                    "coverage": "City politics, investigative work",
                    "reputation": "Liberal mainstream",
                    "influence": "Still matters for older voters",
                    "digital": "Paywalled but reaching younger audience"
                },
                "hidden_influences": {
                    "Key_reporters": "Some receive selective tips",
                    "Editorial_board": "Two members Garden-adjacent",
                    "Story_burial": "Certain investigations stall",
                    "Source_protection": "Absolute for network sources"
                },
                "recent_patterns": [
                    "Soft coverage of certain politicians",
                    "Aggressive pursuit of specific scandals",
                    "Women's leadership features increasing",
                    "Crime statistics contextualized carefully"
                ],
                "manipulation_methods": [
                    "Strategic leaks timed perfectly",
                    "Access journalism rewards",
                    "Social pressure on publishers",
                    "Advertising leverage"
                ]
            },
            {
                "outlet": "KQED (NPR affiliate)",
                "outlet_type": "public_media",
                "structure": "Nonprofit, member-supported",
                "public_trust": {
                    "reputation": "Thoughtful, balanced",
                    "audience": "Educated, progressive",
                    "programs": "Forum, Political Mind, Bay Curious",
                    "influence": "Shapes liberal opinion"
                },
                "subtle_controls": {
                    "Board_members": "Several Garden sympathizers",
                    "Producer_network": "Some understand subtext",
                    "Guest_booking": "Certain experts featured often",
                    "Topic_selection": "Guided by invisible hands"
                },
                "information_flow": [
                    "Careful framing of power issues",
                    "Women's voices elevated systematically",
                    "Alternative justice presented positively",
                    "Certain stories get deep dives, others don't"
                ]
            },
            {
                "outlet": "Mission Local / Berkeleyside",
                "outlet_type": "nonprofit_local_news",
                "model": "Community-funded journalism",
                "authentic_voice": {
                    "coverage": "Hyperlocal, community-focused",
                    "reputation": "Trustworthy, grassroots",
                    "impact": "Disproportionate to size",
                    "funding": "Donations, grants, events"
                },
                "network_relationship": {
                    "Status": "Mostly independent",
                    "Occasional_tips": "From community members",
                    "Protection": "Garden ensures survival",
                    "Useful_for": "Authentic community voice"
                },
                "coverage_patterns": [
                    "Displacement stories highlight victims",
                    "Development coverage skeptical",
                    "Police stories nuanced",
                    "Women leaders featured prominently"
                ]
            },
            {
                "outlet": "Tech Twitter / X",
                "outlet_type": "social_media_discourse",
                "nature": "Chaotic but influential",
                "dynamics": {
                    "Tech_bros": "Loud but vulnerable",
                    "Activists": "Organized and effective",
                    "Journalists": "Break news here first",
                    "Influence_ops": "Hidden in the noise"
                },
                "Garden_operations": {
                    "Sock_puppets": "Boost certain narratives",
                    "Whisper_campaigns": "Destroy reputations",
                    "Honey_traps": "DMs that lead places",
                    "Intelligence": "Who's saying what"
                },
                "recent_operations": [
                    "VC canceled after thread went viral",
                    "Assault story amplified strategically",
                    "Policy proposal memed into existence",
                    "Executive's past excavated precisely"
                ]
            },
            {
                "outlet": "SF Weekly / 48 Hills",
                "outlet_type": "alternative_media",
                "orientation": "Progressive, activist",
                "role": {
                    "coverage": "Left of mainstream",
                    "audience": "Activists, artists, radicals",
                    "influence": "Shapes progressive discourse",
                    "funding": "Precarious but persistent"
                },
                "relationship": {
                    "Editorial": "Sympathetic to Garden goals",
                    "Knowledge": "Suspect but don't probe",
                    "Usefulness": "Amplify certain messages",
                    "Independence": "Maintained carefully"
                }
            },
            {
                "outlet": "Nextdoor",
                "outlet_type": "neighborhood_social",
                "surface_function": "Lost cats and crime panic",
                "hidden_utility": {
                    "Intelligence": "Who thinks what where",
                    "Influence": "Shape neighborhood opinion",
                    "Recruitment": "Identify sympathizers",
                    "Operations": "Coordinate without suspicion"
                },
                "Garden_presence": {
                    "Moderators": "Several neighborhoods",
                    "Narrative_control": "Crime stories contextualized",
                    "Community_building": "Real connections made",
                    "Early_warning": "Problematic people identified"
                }
            }
        ]
    
    @staticmethod
    def get_campaign_finance() -> List[Dict[str, Any]]:
        """How political money flows - publicly and privately"""
        return [
            {
                "funding_type": "Traditional Big Donors",
                "public_face": "Tech executives, real estate, finance",
                "amounts": "$100K+ contributions common",
                "influence_expected": "Access, favorable policy",
                "hidden_reality": {
                    "Some_donors": "Contribute under pressure",
                    "Pressure_types": [
                        "Business complications threatened",
                        "Personal scandals leveraged",
                        "Family concerns raised",
                        "Positive incentives offered"
                    ],
                    "Money_redirected": "To Garden-approved candidates",
                    "Awareness": "Some know they're controlled"
                }
            },
            {
                "funding_type": "Dark Money Groups",
                "public_knowledge": "501c4s with vague names",
                "actual_controllers": "Mix of legitimate and Garden",
                "Garden_fronts": [
                    "Women's Leadership Alliance",
                    "Bay Area Progress Fund",
                    "Citizens for Safe Streets",
                    "Rose Garden Foundation"
                ],
                "activities": [
                    "Attack ads against problems",
                    "Support for allied candidates",
                    "Issue advocacy with subtext",
                    "Voter registration in key areas"
                ],
                "sophistication": "Legally bulletproof structures"
            },
            {
                "funding_type": "Small Dollar Coordination",
                "innovation": "Thousands of small donations",
                "appearance": "Grassroots support",
                "reality": "Some coordinated by network",
                "methods": [
                    "Donation parties at homes",
                    "Workplace campaigns guided",
                    "Social pressure to contribute",
                    "Matching funds from mysterious sources"
                ],
                "effectiveness": "Candidates appear people-powered"
            },
            {
                "funding_type": "In-Kind Contributions",
                "types": [
                    "Volunteer coordination",
                    "Event spaces provided",
                    "Professional services",
                    "Media production"
                ],
                "Garden_specialties": [
                    "Women volunteers who persuade",
                    "Venues that impress donors",
                    "PR that shapes narratives",
                    "Opposition research that finds truth"
                ],
                "value": "Often exceeds cash contributions"
            }
        ]
    
    @staticmethod
    def get_lobbying_influence() -> List[Dict[str, Any]]:
        """How policy gets shaped behind scenes"""
        return [
            {
                "lobby_type": "Traditional Registered Lobbyists",
                "public_players": [
                    "Tech companies",
                    "Real estate interests",
                    "Labor unions",
                    "Business associations"
                ],
                "Garden_infiltration": {
                    "Lobbyists_influenced": "Several major players",
                    "Methods": [
                        "Personal relationships leveraged",
                        "Compromising information held",
                        "Family pressures applied",
                        "Positive rewards offered"
                    ],
                    "Results": "Positions shift subtly",
                    "Awareness": "Some know they're guided"
                }
            },
            {
                "lobby_type": "Social Lobbying",
                "description": "Influence through social connections",
                "Garden_excellence": {
                    "Charity_galas": "Policy made over champagne",
                    "Private_dinners": "Minds changed over wine",
                    "Spa_retreats": "Relaxation and persuasion",
                    "Art_openings": "Culture and politics mix"
                },
                "practitioners": [
                    "Society women with purpose",
                    "Professional wives with agency",
                    "Event planners with agendas",
                    "Hostesses who harvest information"
                ]
            },
            {
                "lobby_type": "Bedroom Lobbying",
                "never_acknowledged": "But everyone knows",
                "Garden_advantage": {
                    "Intelligence": "Who sleeps with whom",
                    "Leverage": "Desires create vulnerability",
                    "Operators": "Skilled in specific persuasion",
                    "Results": "Minds changed through bodies"
                },
                "ethical_lines": [
                    "Consensual but calculated",
                    "Power dynamics complex",
                    "Information gathered carefully",
                    "Blackmail vs influence debated"
                ]
            }
        ]
    
    @staticmethod
    def get_voting_patterns() -> List[Dict[str, Any]]:
        """How elections really work in SF"""
        return [
            {
                "pattern_name": "The Mission Vote",
                "demographic": "Latino community, artists, activists",
                "traditional_behavior": "Progressive, low turnout",
                "Garden_influence": {
                    "Registration_drives": "Targeted effectively",
                    "Trusted_messengers": "Community women leaders",
                    "Issues_emphasized": "Protection and preservation",
                    "Turnout_operations": "Coordinated precisely"
                },
                "swing_potential": "Can decide close elections"
            },
            {
                "pattern_name": "Pacific Heights Persuasion",
                "demographic": "Old money, society families",
                "traditional_behavior": "Moderate, high turnout",
                "influence_methods": [
                    "Social pressure at clubs",
                    "Charity board discussions",
                    "Spouse influence campaigns",
                    "Reputation considerations"
                ],
                "recent_shifts": "More progressive on women's issues"
            },
            {
                "pattern_name": "Tech Worker Ambivalence",
                "demographic": "Young, transplant, busy",
                "traditional_behavior": "Liberal but low turnout",
                "engagement_strategies": [
                    "Apps and digital outreach",
                    "Workplace peer pressure",
                    "Issues framed as disruption",
                    "Dating app political messaging"
                ],
                "vulnerability": "Easily influenced if engaged"
            },
            {
                "pattern_name": "Service Worker Solidarity",
                "demographic": "Hotel, restaurant, gig workers",
                "organizing_reality": {
                    "Union_influenced": "But not monolithic",
                    "Garden_connections": "Through women leaders",
                    "Issues": "Housing, wages, dignity",
                    "Mobilization": "When properly motivated"
                },
                "untapped_power": "Could reshape city politics"
            }
        ]
    
    @staticmethod
    def get_political_scandals() -> List[Dict[str, Any]]:
        """Recent and brewing political scandals"""
        return [
            {
                "scandal_name": "The Supervisor's Affair",
                "public_version": "Consensual relationship exposed",
                "actual_story": {
                    "Reality": "Predatory behavior pattern",
                    "Garden_involvement": "Victim protected and supported",
                    "Evidence": "Selectively released",
                    "Outcome": "Resignation with prejudice"
                },
                "lessons_taught": [
                    "Power has consequences",
                    "Networks protect their own",
                    "Alternative justice works",
                    "Fear spreads efficiently"
                ]
            },
            {
                "scandal_name": "Port Commissioner's Finances",
                "status": "Brewing but not yet public",
                "known_by": {
                    "Feds": "Investigating slowly",
                    "Media": "Sniffing around edges",
                    "Garden": "Complete picture held",
                    "Commissioner": "Sweating profusely"
                },
                "strategic_options": [
                    "Full exposure destroys trafficking",
                    "Partial leak maintains leverage",
                    "Quiet resolution serves justice",
                    "Continued observation gathers intelligence"
                ],
                "Queen's_decision": "Pending"
            },
            {
                "scandal_name": "The Mayor's Chief of Staff",
                "nature": "Compromising photographs",
                "twist": "Consensual BDSM presented as scandal",
                "Garden_response": {
                    "Protection": "Narrative controlled",
                    "Counter": "Accusers' histories exposed",
                    "Result": "Attempt backfired spectacularly",
                    "Message": "We protect our own"
                },
                "aftermath": "Chief of Staff now deeper ally"
            },
            {
                "scandal_name": "Campaign Finance Violations",
                "target": "Multiple candidates",
                "evidence_held": "By Garden operatives",
                "strategic_use": [
                    "Keep as insurance",
                    "Deploy if candidate strays",
                    "Trade for cooperation",
                    "Mutual assured destruction"
                ],
                "current_status": "Multiple swords hanging"
            }
        ]
    
    @staticmethod
    def get_policy_battles() -> List[Dict[str, Any]]:
        """Current policy fights with hidden dimensions"""
        return [
            {
                "policy_area": "Sex Work Decriminalization",
                "public_debate": {
                    "Pro": "Safety, rights, dignity",
                    "Con": "Neighborhood impacts, trafficking",
                    "Politics": "Progressive split"
                },
                "Garden_position": {
                    "Official": "Neutral publicly",
                    "Reality": "Complex interests",
                    "Concerns": [
                        "Protects consensual workers",
                        "But enables traffickers",
                        "Visibility vs safety debate",
                        "Control vs chaos"
                    ],
                    "Strategy": "Shape implementation details"
                },
                "behind_scenes": [
                    "Worker organizations influenced",
                    "Specific protections inserted",
                    "Enforcement priorities guided",
                    "Bad actors identified for removal"
                ]
            },
            {
                "policy_area": "Surveillance Technology",
                "public_concern": "Privacy vs safety",
                "tech_company_position": "Innovation and efficiency",
                "hidden_battle": {
                    "Garden_needs": "Some surveillance useful",
                    "But": "Can't be turned against network",
                    "Solution": "Backdoors and blind spots",
                    "Implementation": "Allies in tech compliance"
                },
                "specific_fights": [
                    "Facial recognition limits",
                    "Data retention policies",
                    "Access controls",
                    "Audit requirements"
                ]
            },
            {
                "policy_area": "Zoning Reform",
                "surface_issue": "Housing production",
                "deeper_implications": {
                    "Which_communities": "Protected or sacrificed",
                    "Power_structures": "Preserved or disrupted",
                    "Underground_venues": "Threatened or secured",
                    "Networks": "Maintained or scattered"
                },
                "Garden_operations": [
                    "Specific parcels protected",
                    "Certain developers blocked",
                    "Community input amplified",
                    "Alternative proposals funded"
                ]
            }
        ]
    
    @staticmethod
    def get_enforcement_realities() -> List[Dict[str, Any]]:
        """How laws actually get enforced"""
        return [
            {
                "enforcement_area": "Prostitution/Sex Work",
                "official_policy": "Still illegal but deprioritized",
                "street_reality": {
                    "Consensual_adult": "Generally ignored",
                    "Street_level": "Occasional sweeps",
                    "High_end": "Never touched",
                    "Trafficking": "Aggressively pursued"
                },
                "Garden_influence": [
                    "Intel on actual trafficking",
                    "Protection for consensual workers",
                    "Bad date lists maintained",
                    "Predators targeted for exposure"
                ],
                "cop_awareness": "Vice knows the deal"
            },
            {
                "enforcement_area": "Drug Policy",
                "public_approach": "Harm reduction",
                "selective_enforcement": {
                    "Users": "Services offered",
                    "Small_dealers": "Depends on behavior",
                    "Major_dealers": "If they harm community",
                    "Pharma_bros": "Sudden scrutiny"
                },
                "Garden_interest": "Protect vulnerable users",
                "methods": [
                    "Bad dealer information provided",
                    "Treatment slots appear for some",
                    "Predatory dealers vanish",
                    "Corrupt cops exposed"
                ]
            },
            {
                "enforcement_area": "White Collar Crime",
                "usual_pattern": "Rarely prosecuted",
                "when_Garden_interested": {
                    "Evidence": "Appears perfectly packaged",
                    "Witnesses": "Suddenly cooperative",
                    "Media": "Sustained coverage",
                    "Result": "Actual consequences"
                },
                "recent_examples": [
                    "Tech executive wage theft exposed",
                    "Landlord fraud prosecution",
                    "Investor sexual assault case",
                    "Corporate espionage revealed"
                ]
            }
        ]
    
    @staticmethod
    def get_political_alliances() -> List[Dict[str, Any]]:
        """Hidden alliances that shape politics"""
        return [
            {
                "alliance_name": "The Progressive Feminists",
                "public_face": "Women's rights advocates",
                "deeper_reality": "Garden entry point",
                "members_include": [
                    "Elected officials",
                    "Nonprofit leaders",
                    "Union organizers",
                    "Academic feminists"
                ],
                "coordination": [
                    "Monthly strategy sessions",
                    "Encrypted communications",
                    "Resource sharing",
                    "Candidate vetting"
                ],
                "power_demonstrated": "Swing close elections"
            },
            {
                "alliance_name": "The Quiet Money",
                "composition": "Old SF families and new female wealth",
                "public_activity": "Philanthropy and galas",
                "actual_function": [
                    "Fund Garden operations",
                    "Buy political influence",
                    "Protect class interests",
                    "Shape city development"
                ],
                "coordination_method": "Garden parties and salons",
                "recent_victory": "Killed waterfront development"
            },
            {
                "alliance_name": "The Service Sector Sisters",
                "membership": "Women in hospitality, healthcare, sex work",
                "public_goals": "Worker rights and dignity",
                "hidden_strength": [
                    "Information network supreme",
                    "Access to powerful men",
                    "Coordination across industries",
                    "Protection for vulnerable"
                ],
                "Garden_connection": "Direct recruitment pipeline",
                "political_impact": "Living wage ordinances"
            }
        ]
    
    @staticmethod
    def get_political_pressure_points() -> List[Dict[str, Any]]:
        """Where and how pressure gets applied"""
        return [
            {
                "pressure_type": "Personal Scandals",
                "description": "Everyone has secrets in SF",
                "Garden_advantage": {
                    "Intelligence_network": "Knows most secrets",
                    "Documentation": "Evidence preserved carefully",
                    "Deployment": "Strategic and devastating",
                    "Mercy": "Available for cooperation"
                },
                "recent_uses": [
                    "Supervisor changed vote on development",
                    "Commissioner resigned suddenly",
                    "CEO became generous donor",
                    "Judge reversed concerning pattern"
                ]
            },
            {
                "pressure_type": "Family Concerns",
                "description": "Threats to loved ones motivate",
                "ethical_approach": {
                    "Never_harm": "Family off limits",
                    "But_inform": "About children's activities",
                    "Or_offer": "Help with family problems",
                    "Creating": "Gratitude or fear"
                },
                "examples": [
                    "Daughter's college admission helped",
                    "Son's drug problem addressed",
                    "Spouse's affair revealed carefully",
                    "Parent's medical care arranged"
                ]
            },
            {
                "pressure_type": "Financial Complications",
                "methods": [
                    "Investments suddenly problematic",
                    "Loans called in unexpectedly",
                    "Tax issues emerge mysteriously",
                    "Donors coordinate withdrawal"
                ],
                "Garden_capability": "Extensive through network",
                "balanced_by": "Rewards for cooperation"
            },
            {
                "pressure_type": "Professional Destruction",
                "nuclear_option": "Career-ending revelations",
                "stages": [
                    "Warning shots fired",
                    "Escalating complications",
                    "Final ultimatum given",
                    "Total destruction if needed"
                ],
                "effectiveness": "Fear often sufficient"
            }
        ]
    
    @staticmethod
    def get_quest_hooks() -> List[Dict[str, Any]]:
        """Political storylines for players"""
        return [
            {
                "quest_name": "The Blackmail Files",
                "quest_type": "investigation_thriller",
                "setup": {
                    "situation": "Encrypted drives with political dirt surface",
                    "question": "Garden's files or someone else's?",
                    "stakes": "Could reshape city power structure",
                    "urgency": "Multiple parties hunting"
                },
                "player_entry": [
                    "Hired to find drives",
                    "Accidentally obtain one",
                    "Target of contents",
                    "Garden operative assigned"
                ],
                "complications": [
                    "Some files authentic, others false",
                    "Federal investigation triggered",
                    "Media circus erupting",
                    "Violence escalating"
                ],
                "possible_outcomes": [
                    "Destroy files, preserve stability",
                    "Selective release for justice",
                    "Full exposure, chaos ensues",
                    "Trade to Garden for protection"
                ]
            },
            {
                "quest_name": "The Recall Campaign",
                "quest_type": "political_intrigue",
                "scenario": {
                    "target": "Progressive official Garden protects",
                    "attackers": "Coalition with hidden backing",
                    "public_issue": "Crime and competence",
                    "real_issue": "Official knows too much"
                },
                "player_roles": [
                    "Campaign strategist",
                    "Opposition researcher",
                    "Media manipulator",
                    "Garden coordinator"
                ],
                "dirty_tricks": [
                    "Fake evidence planted",
                    "Witnesses intimidated",
                    "Donors pressured",
                    "Votes suppressed"
                ],
                "resolution_paths": [
                    "Win through superior tactics",
                    "Expose opposition corruption",
                    "Negotiate behind scenes",
                    "Let recall succeed strategically"
                ]
            },
            {
                "quest_name": "The Port Investigation",
                "quest_type": "corruption_expose",
                "setup": {
                    "surface": "Federal trafficking investigation",
                    "reality": "Network of complicity",
                    "Garden_position": "Conflicted interests",
                    "player_position": "Dangerous middle"
                },
                "key_players": [
                    "Commissioner Huang - playing all sides",
                    "Federal agents - some corrupted",
                    "Trafficking rings - multiple competing",
                    "Garden operatives - saving victims"
                ],
                "investigation_threads": [
                    "Follow the money",
                    "Track the shipments",
                    "Identify the victims",
                    "Expose the protectors"
                ],
                "moral_complexities": [
                    "Saving victims vs preserving network",
                    "Justice vs larger mission",
                    "Truth vs useful lies",
                    "Law vs higher justice"
                ]
            },
            {
                "quest_name": "The Succession Question",
                "quest_type": "power_struggle",
                "situation": {
                    "opening": "Mayor announces retirement",
                    "candidates": "Multiple with hidden ties",
                    "Garden_goal": "Install sympathetic leader",
                    "complications": "Competing factions within"
                },
                "player_involvement": [
                    "Kingmaker role",
                    "Candidate yourself",
                    "Opposition researcher",
                    "Network coordinator"
                ],
                "campaign_elements": [
                    "Building coalitions",
                    "Neutralizing threats",
                    "Managing scandals",
                    "Controlling narrative"
                ],
                "deeper_game": [
                    "Who controls the Garden?",
                    "Network evolution debate",
                    "Generational power transfer",
                    "Future of the shadow city"
                ]
            }
        ]
    
    @staticmethod
    def get_conflict_news_cycles() -> List[Dict[str, Any]]:
        """How different media covers political events"""
        return [
            {
                "event": "Supervisor Sex Scandal",
                "date": "July 2025",
                "chronicle_headline": "Supervisor Martinez Resigns Amid Allegations",
                "chronicle_angle": {
                    "focus": "Political implications",
                    "sources": "Official statements, colleagues",
                    "tone": "Measured, both-sides",
                    "buried": "Pattern of behavior details"
                },
                "mission_local_headline": "Community Demands Justice for Martinez Victims",
                "mission_local_angle": {
                    "focus": "Victim voices, community impact",
                    "sources": "Neighbors, local organizations",
                    "tone": "Angry, demanding action",
                    "highlighted": "Power abuse patterns"
                },
                "twitter_narrative": {
                    "trending": "#MartinezMustGo #BelieveWomen",
                    "factions": [
                        "Defenders citing conspiracy",
                        "Victims sharing stories",
                        "Political opportunists",
                        "Garden amplifiers"
                    ],
                    "manipulation": "Key threads boosted/buried"
                },
                "kqed_coverage": {
                    "approach": "Systemic analysis",
                    "experts": "Gender studies professors, psychologists",
                    "frame": "Power and accountability",
                    "subtle_message": "This is why we need women leaders"
                },
                "actual_story": {
                    "reality": "Serial predator finally caught",
                    "Garden_role": "Protected victims, provided evidence",
                    "timing": "Strategic for political goals",
                    "aftermath": "Replacement more amenable"
                }
            },
            {
                "event": "Housing Bond Measure Passage",
                "date": "November 2025",
                "chronicle_headline": "Voters Approve $2B Housing Bond",
                "chronicle_angle": {
                    "focus": "Economic impact, tax implications",
                    "sources": "City economists, builders",
                    "buried": "Specific beneficiary communities",
                    "editorial": "Cautious support"
                },
                "48_hills_headline": "People Power Wins Housing Victory",
                "48_hills_angle": {
                    "focus": "Grassroots organizing triumph",
                    "sources": "Tenant organizers, activists",
                    "highlighted": "Corporate opposition defeated",
                    "analysis": "Step toward housing justice"
                },
                "tech_twitter_reaction": {
                    "VCs": "Complaining about taxes",
                    "Workers": "Cautiously optimistic",
                    "Urbanists": "Not enough but progress",
                    "Hidden": "Coordinated supportive messaging"
                },
                "garden_perspective": {
                    "public_win": "Affordable housing funded",
                    "hidden_win": "Specific provisions inserted",
                    "network_benefit": "Control over site selection",
                    "long_game": "Communities protected from displacement"
                }
            },
            {
                "event": "DA Recall Election",
                "date": "Ongoing coverage",
                "chronicle_position": "Studiously neutral with lean toward recall",
                "examiner_position": "Strongly pro-recall",
                "progressive_media": "Anti-recall but acknowledging problems",
                "social_media_wars": {
                    "hashtags": "#RecallChesa vs #StandWithChesa",
                    "bot_activity": "High on both sides",
                    "influencer_positions": "Carefully orchestrated",
                    "Garden_strategy": "Chaos and confusion"
                },
                "leaked_information": {
                    "timing": "Strategic drips",
                    "content": "Damaging to both sides",
                    "source": "Officially unknown",
                    "effect": "Keeps race tight"
                },
                "hidden_agenda": {
                    "Garden_goal": "Controllable DA either way",
                    "Methods": "Leverage on both candidates",
                    "Backup_plan": "Influence key assistants",
                    "Real_victor": "Shadow justice system"
                }
            }
        ]
