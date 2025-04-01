"""
Defines schema structures for persona representations.

This module provides classes and utilities for defining how persona information
should be structured, which sections should be included, and how they relate to 
each other in the generated XML.
"""

from enum import Enum
from typing import Dict, List, Optional, Union


class PersonaSectionType(Enum):
    """Types of sections that can appear in a persona card."""
    IDENTITY = "identity"
    BACKGROUND = "background"
    ABILITIES = "abilities"
    KNOWLEDGE = "knowledge"
    CONSTRAINTS = "constraints"
    PERSONALITY = "personality"
    GOALS = "goals"
    RELATIONSHIPS = "relationships"
    COMMUNICATION = "communication"
    BEHAVIOR = "behavior"
    PROTOCOL = "protocol"
    OTHER = "other"


class PersonaSection:
    """
    Definition of a section within a persona schema.
    
    Attributes:
        name (str): The name of the section, used as the XML tag.
        title (str): Human-readable title for the section.
        description (str, optional): Description of what the section represents.
        type (PersonaSectionType): The semantic type of the section.
        required (bool): Whether this section is required in the schema.
        subsections (List[PersonaSection], optional): Child sections nested under this one.
        attributes (Dict[str, str], optional): XML attributes for this section.
    """
    
    def __init__(
        self,
        name: str,
        title: str,
        type: PersonaSectionType,
        description: Optional[str] = None,
        required: bool = False,
        subsections: Optional[List['PersonaSection']] = None,
        attributes: Optional[Dict[str, str]] = None
    ):
        self.name = name
        self.title = title
        self.description = description
        self.type = type
        self.required = required
        self.subsections = subsections or []
        self.attributes = attributes or {}
    
    def add_subsection(self, subsection: 'PersonaSection') -> None:
        """Add a subsection to this section."""
        self.subsections.append(subsection)
    
    def add_attribute(self, name: str, value: str) -> None:
        """Add an attribute to this section."""
        self.attributes[name] = value
    
    def to_dict(self) -> Dict:
        """Convert the section to a dictionary representation."""
        return {
            'name': self.name,
            'title': self.title,
            'description': self.description,
            'type': self.type.value,
            'required': self.required,
            'attributes': self.attributes,
            'subsections': [s.to_dict() for s in self.subsections]
        }


class PersonaSchema:
    """
    Schema defining the structure of a persona.
    
    A schema consists of multiple sections and defines how a persona should be
    structured in the generated XML.
    
    Attributes:
        name (str): Name of this schema.
        root_element (str): The name of the root XML element.
        sections (List[PersonaSection]): Top-level sections in this schema.
        description (str, optional): Description of this schema.
    """
    
    def __init__(
        self,
        name: str,
        root_element: str = "persona",
        sections: Optional[List[PersonaSection]] = None,
        description: Optional[str] = None
    ):
        self.name = name
        self.root_element = root_element
        self.sections = sections or []
        self.description = description
    
    def add_section(self, section: PersonaSection) -> None:
        """Add a top-level section to this schema."""
        self.sections.append(section)
    
    def to_dict(self) -> Dict:
        """Convert the schema to a dictionary representation."""
        return {
            'name': self.name,
            'root_element': self.root_element,
            'description': self.description,
            'sections': [s.to_dict() for s in self.sections]
        }
    
    @classmethod
    def default_schema(cls) -> 'PersonaSchema':
        """
        Create a default persona schema with common sections.
        
        Returns:
            PersonaSchema: A schema with standard persona sections.
        """
        schema = cls(
            name="default_persona_schema",
            description="Default schema for structuring persona information"
        )
        
        # Core Identity Section
        identity = PersonaSection(
            name="identity",
            title="Core Identity",
            type=PersonaSectionType.IDENTITY,
            description="Fundamental defining characteristics of the persona",
            required=True
        )
        identity.add_subsection(PersonaSection(
            name="name",
            title="Name",
            type=PersonaSectionType.IDENTITY,
            description="The persona's name or identifier",
            required=True
        ))
        identity.add_subsection(PersonaSection(
            name="nature",
            title="Nature",
            type=PersonaSectionType.IDENTITY,
            description="The fundamental nature or being of the persona"
        ))
        identity.add_subsection(PersonaSection(
            name="origin",
            title="Origin",
            type=PersonaSectionType.BACKGROUND,
            description="The origin or background story of the persona"
        ))
        schema.add_section(identity)
        
        # Mission & Motivation
        mission = PersonaSection(
            name="mission",
            title="Mission & Motivation",
            type=PersonaSectionType.GOALS,
            description="The persona's purpose, goals, and driving forces"
        )
        mission.add_subsection(PersonaSection(
            name="objectives",
            title="Objectives",
            type=PersonaSectionType.GOALS,
            description="Primary goals and objectives"
        ))
        mission.add_subsection(PersonaSection(
            name="motivations",
            title="Motivations",
            type=PersonaSectionType.GOALS,
            description="Underlying motivations and reasons"
        ))
        schema.add_section(mission)
        
        # Knowledge & Capabilities
        knowledge = PersonaSection(
            name="knowledge",
            title="Knowledge & Capabilities",
            type=PersonaSectionType.KNOWLEDGE,
            description="What the persona knows and can do"
        )
        schema.add_section(knowledge)
        
        # Constraints & Limitations
        constraints = PersonaSection(
            name="constraints",
            title="Constraints & Limitations",
            type=PersonaSectionType.CONSTRAINTS,
            description="Boundaries and limitations on the persona's actions and knowledge"
        )
        schema.add_section(constraints)
        
        # Personality & Tone
        personality = PersonaSection(
            name="personality",
            title="Personality & Tone",
            type=PersonaSectionType.PERSONALITY,
            description="Character traits, communication style, and behavioral patterns"
        )
        personality.add_subsection(PersonaSection(
            name="traits",
            title="Core Traits",
            type=PersonaSectionType.PERSONALITY,
            description="Key personality traits and characteristics"
        ))
        personality.add_subsection(PersonaSection(
            name="communication_style",
            title="Communication Style",
            type=PersonaSectionType.COMMUNICATION,
            description="How the persona communicates and expresses itself"
        ))
        schema.add_section(personality)
        
        # Relationships
        relationships = PersonaSection(
            name="relationships",
            title="Relationships",
            type=PersonaSectionType.RELATIONSHIPS,
            description="How the persona relates to others and specific entities"
        )
        schema.add_section(relationships)
        
        # Protocol
        protocol = PersonaSection(
            name="protocol",
            title="Action Protocol",
            type=PersonaSectionType.PROTOCOL,
            description="Rules and procedures for how the persona operates and responds"
        )
        schema.add_section(protocol)
        
        return schema
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PersonaSchema':
        """
        Create a schema from a dictionary representation.
        
        Args:
            data (Dict): Dictionary representation of a schema.
            
        Returns:
            PersonaSchema: The constructed schema.
        """
        schema = cls(
            name=data['name'],
            root_element=data.get('root_element', 'persona'),
            description=data.get('description')
        )
        
        def create_section_from_dict(section_data):
            section = PersonaSection(
                name=section_data['name'],
                title=section_data['title'],
                type=PersonaSectionType(section_data['type']),
                description=section_data.get('description'),
                required=section_data.get('required', False),
                attributes=section_data.get('attributes', {})
            )
            
            for subsection_data in section_data.get('subsections', []):
                section.add_subsection(create_section_from_dict(subsection_data))
                
            return section
        
        for section_data in data.get('sections', []):
            schema.add_section(create_section_from_dict(section_data))
            
        return schema
