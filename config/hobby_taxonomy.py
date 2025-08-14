"""
Hobby Taxonomy and Career Hierarchy Configuration
================================================

This module defines the relationship between hobbies and careers,
as well as the hierarchical structure of career classifications.
"""

# Career hierarchy mapping
CAREER_HIERARCHY = {
    'software_engineer': {
        'field': 'engineering',
        'broad_category': 'STEM',
        'skills': ['programming', 'problem_solving', 'technical_analysis']
    },
    'data_scientist': {
        'field': 'data_science',
        'broad_category': 'STEM',
        'skills': ['data_analysis', 'statistics', 'programming', 'research']
    },
    'web_developer': {
        'field': 'engineering',
        'broad_category': 'STEM',
        'skills': ['programming', 'web_design', 'technical_skills']
    },
    'mechanical_engineer': {
        'field': 'engineering',
        'broad_category': 'STEM',
        'skills': ['engineering_design', 'problem_solving', 'technical_analysis']
    },
    'financial_analyst': {
        'field': 'finance',
        'broad_category': 'Business',
        'skills': ['analytical_thinking', 'data_analysis', 'financial_modeling']
    },
    'marketing_manager': {
        'field': 'marketing',
        'broad_category': 'Business',
        'skills': ['communication', 'creativity', 'leadership', 'strategic_thinking']
    },
    'accountant': {
        'field': 'finance',
        'broad_category': 'Business',
        'skills': ['analytical_thinking', 'attention_to_detail', 'numerical_skills']
    },
    'graphic_designer': {
        'field': 'design',
        'broad_category': 'Creative',
        'skills': ['creativity', 'visual_design', 'artistic_skills']
    },
    'ux_designer': {
        'field': 'design',
        'broad_category': 'Creative',
        'skills': ['creativity', 'user_research', 'problem_solving', 'technical_skills']
    },
    'chef': {
        'field': 'culinary',
        'broad_category': 'Creative',
        'skills': ['creativity', 'cooking', 'attention_to_detail', 'time_management']
    },
    'teacher': {
        'field': 'education',
        'broad_category': 'Social',
        'skills': ['communication', 'patience', 'leadership', 'empathy']
    },
    'psychologist': {
        'field': 'psychology',
        'broad_category': 'Healthcare',
        'skills': ['empathy', 'analytical_thinking', 'communication', 'research']
    },
    'doctor': {
        'field': 'medicine',
        'broad_category': 'Healthcare',
        'skills': ['analytical_thinking', 'empathy', 'attention_to_detail', 'problem_solving']
    },
    'nurse': {
        'field': 'healthcare',
        'broad_category': 'Healthcare',
        'skills': ['empathy', 'attention_to_detail', 'communication', 'teamwork']
    },
    'lawyer': {
        'field': 'legal',
        'broad_category': 'Law_Government',
        'skills': ['analytical_thinking', 'communication', 'research', 'attention_to_detail']
    }
}

# Hobby taxonomy with career relevance scores
HOBBY_TAXONOMY = {
    'programming': {
        'category': 'technical',
        'skills': ['logical_thinking', 'problem_solving', 'technical_skills'],
        'career_relevance': {
            'software_engineer': 0.95,
            'data_scientist': 0.85,
            'web_developer': 0.90,
            'mechanical_engineer': 0.30,
            'financial_analyst': 0.40,
            'marketing_manager': 0.20,
            'accountant': 0.25,
            'graphic_designer': 0.35,
            'ux_designer': 0.70,
            'chef': 0.10,
            'teacher': 0.25,
            'psychologist': 0.20,
            'doctor': 0.15,
            'nurse': 0.15,
            'lawyer': 0.20
        }
    },
    'research': {
        'category': 'analytical',
        'skills': ['analytical_thinking', 'attention_to_detail', 'problem_solving'],
        'career_relevance': {
            'software_engineer': 0.60,
            'data_scientist': 0.90,
            'web_developer': 0.40,
            'mechanical_engineer': 0.70,
            'financial_analyst': 0.80,
            'marketing_manager': 0.60,
            'accountant': 0.50,
            'graphic_designer': 0.30,
            'ux_designer': 0.75,
            'chef': 0.20,
            'teacher': 0.60,
            'psychologist': 0.95,
            'doctor': 0.85,
            'nurse': 0.40,
            'lawyer': 0.80
        }
    },
    'writing': {
        'category': 'creative',
        'skills': ['creativity', 'communication', 'language_skills'],
        'career_relevance': {
            'software_engineer': 0.30,
            'data_scientist': 0.50,
            'web_developer': 0.40,
            'mechanical_engineer': 0.30,
            'financial_analyst': 0.45,
            'marketing_manager': 0.85,
            'accountant': 0.30,
            'graphic_designer': 0.60,
            'ux_designer': 0.65,
            'chef': 0.25,
            'teacher': 0.80,
            'psychologist': 0.70,
            'doctor': 0.40,
            'nurse': 0.35,
            'lawyer': 0.90
        }
    },
    'music': {
        'category': 'creative',
        'skills': ['creativity', 'artistic_expression', 'discipline'],
        'career_relevance': {
            'software_engineer': 0.20,
            'data_scientist': 0.25,
            'web_developer': 0.30,
            'mechanical_engineer': 0.20,
            'financial_analyst': 0.15,
            'marketing_manager': 0.50,
            'accountant': 0.15,
            'graphic_designer': 0.70,
            'ux_designer': 0.55,
            'chef': 0.40,
            'teacher': 0.60,
            'psychologist': 0.35,
            'doctor': 0.20,
            'nurse': 0.25,
            'lawyer': 0.25
        }
    },
    'photography': {
        'category': 'creative',
        'skills': ['creativity', 'visual_composition', 'technical_skills'],
        'career_relevance': {
            'software_engineer': 0.25,
            'data_scientist': 0.30,
            'web_developer': 0.50,
            'mechanical_engineer': 0.25,
            'financial_analyst': 0.20,
            'marketing_manager': 0.70,
            'accountant': 0.15,
            'graphic_designer': 0.85,
            'ux_designer': 0.75,
            'chef': 0.45,
            'teacher': 0.40,
            'psychologist': 0.30,
            'doctor': 0.20,
            'nurse': 0.20,
            'lawyer': 0.25
        }
    },
    'cooking': {
        'category': 'creative',
        'skills': ['creativity', 'attention_to_detail', 'time_management'],
        'career_relevance': {
            'software_engineer': 0.15,
            'data_scientist': 0.20,
            'web_developer': 0.20,
            'mechanical_engineer': 0.25,
            'financial_analyst': 0.15,
            'marketing_manager': 0.30,
            'accountant': 0.15,
            'graphic_designer': 0.40,
            'ux_designer': 0.35,
            'chef': 0.95,
            'teacher': 0.30,
            'psychologist': 0.25,
            'doctor': 0.20,
            'nurse': 0.25,
            'lawyer': 0.20
        }
    },
    'volunteering': {
        'category': 'social',
        'skills': ['empathy', 'teamwork', 'communication'],
        'career_relevance': {
            'software_engineer': 0.30,
            'data_scientist': 0.35,
            'web_developer': 0.35,
            'mechanical_engineer': 0.40,
            'financial_analyst': 0.35,
            'marketing_manager': 0.70,
            'accountant': 0.30,
            'graphic_designer': 0.40,
            'ux_designer': 0.60,
            'chef': 0.45,
            'teacher': 0.85,
            'psychologist': 0.80,
            'doctor': 0.75,
            'nurse': 0.90,
            'lawyer': 0.50
        }
    },
    'public_speaking': {
        'category': 'social',
        'skills': ['communication', 'confidence', 'leadership'],
        'career_relevance': {
            'software_engineer': 0.40,
            'data_scientist': 0.50,
            'web_developer': 0.45,
            'mechanical_engineer': 0.50,
            'financial_analyst': 0.60,
            'marketing_manager': 0.90,
            'accountant': 0.40,
            'graphic_designer': 0.50,
            'ux_designer': 0.65,
            'chef': 0.35,
            'teacher': 0.95,
            'psychologist': 0.70,
            'doctor': 0.60,
            'nurse': 0.50,
            'lawyer': 0.85
        }
    },
    'sports': {
        'category': 'physical',
        'skills': ['teamwork', 'discipline', 'goal_orientation'],
        'career_relevance': {
            'software_engineer': 0.35,
            'data_scientist': 0.40,
            'web_developer': 0.35,
            'mechanical_engineer': 0.45,
            'financial_analyst': 0.40,
            'marketing_manager': 0.60,
            'accountant': 0.35,
            'graphic_designer': 0.40,
            'ux_designer': 0.45,
            'chef': 0.50,
            'teacher': 0.65,
            'psychologist': 0.45,
            'doctor': 0.50,
            'nurse': 0.55,
            'lawyer': 0.45
        }
    },
    'reading': {
        'category': 'intellectual',
        'skills': ['analytical_thinking', 'knowledge_acquisition', 'focus'],
        'career_relevance': {
            'software_engineer': 0.50,
            'data_scientist': 0.70,
            'web_developer': 0.45,
            'mechanical_engineer': 0.55,
            'financial_analyst': 0.65,
            'marketing_manager': 0.60,
            'accountant': 0.50,
            'graphic_designer': 0.45,
            'ux_designer': 0.60,
            'chef': 0.30,
            'teacher': 0.80,
            'psychologist': 0.85,
            'doctor': 0.75,
            'nurse': 0.50,
            'lawyer': 0.90
        }
    },
    'investing': {
        'category': 'analytical',
        'skills': ['analytical_thinking', 'risk_assessment', 'strategic_planning'],
        'career_relevance': {
            'software_engineer': 0.40,
            'data_scientist': 0.60,
            'web_developer': 0.35,
            'mechanical_engineer': 0.35,
            'financial_analyst': 0.95,
            'marketing_manager': 0.50,
            'accountant': 0.80,
            'graphic_designer': 0.25,
            'ux_designer': 0.30,
            'chef': 0.20,
            'teacher': 0.30,
            'psychologist': 0.35,
            'doctor': 0.40,
            'nurse': 0.25,
            'lawyer': 0.60
        }
    }
}

# Skill categories and their descriptions
SKILL_CATEGORIES = {
    'technical': {
        'description': 'Skills related to technology, programming, and technical problem-solving',
        'examples': ['programming', 'technical_analysis', 'system_design']
    },
    'creative': {
        'description': 'Skills related to artistic expression, design, and innovation',
        'examples': ['artistic_expression', 'visual_design', 'creativity']
    },
    'analytical': {
        'description': 'Skills related to data analysis, research, and logical reasoning',
        'examples': ['analytical_thinking', 'data_analysis', 'research']
    },
    'social': {
        'description': 'Skills related to interpersonal communication and teamwork',
        'examples': ['communication', 'empathy', 'leadership']
    },
    'physical': {
        'description': 'Skills related to physical activities and coordination',
        'examples': ['physical_coordination', 'endurance', 'teamwork']
    },
    'intellectual': {
        'description': 'Skills related to learning, knowledge acquisition, and mental processes',
        'examples': ['knowledge_acquisition', 'focus', 'memory']
    }
}

# Helper functions
def get_career_field(career):
    """Get the field for a given career"""
    if career not in CAREER_HIERARCHY:
        raise KeyError(f"Career '{career}' not found in hierarchy")
    return CAREER_HIERARCHY[career]['field']

def get_career_broad_category(career):
    """Get the broad category for a given career"""
    if career not in CAREER_HIERARCHY:
        raise KeyError(f"Career '{career}' not found in hierarchy")
    return CAREER_HIERARCHY[career]['broad_category']

def get_hobby_relevance(hobby, career):
    """Get the relevance score of a hobby for a specific career"""
    if hobby not in HOBBY_TAXONOMY:
        raise KeyError(f"Hobby '{hobby}' not found in taxonomy")
        
    if career not in HOBBY_TAXONOMY[hobby]['career_relevance']:
        raise KeyError(f"Career '{career}' not found in hobby '{hobby}' relevance mapping")
        
    return HOBBY_TAXONOMY[hobby]['career_relevance'][career]

def get_all_careers():
    """Get list of all available careers"""
    return list(CAREER_HIERARCHY.keys())

def get_all_hobbies():
    """Get list of all available hobbies"""
    return list(HOBBY_TAXONOMY.keys())
