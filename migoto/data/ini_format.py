from dataclasses import dataclass, field

@dataclass
class INI_Line:
    """Class to represent a line in an ini file"""

    key: str
    value: str
    is_value_pair: bool
    _stripped_lower_key: str = field(init=False, repr=False)
    _stripped_lower_value: str = field(init=False, repr=False)

    def __setattr__(self, name, value) -> None:
        """Override the __setattr__ method to strip and lowercase the key and value"""
        if name == "key":
            self._stripped_lower_key = value.strip().lower()
        elif name == "value":
            self._stripped_lower_value = value.strip().lower()
        super().__setattr__(name, value)

    def has_key(self, key: str) -> bool:
        """Check if the line has a specific key"""
        return self._stripped_lower_key == key.strip().lower()

    def key_startswith(self, key: str) -> bool:
        """Check if the line key starts with a specific string"""
        return self._stripped_lower_key.startswith(key.strip().lower())


@dataclass
class Section:
    """Class to represent a section in an ini file"""

    name: str
    lines: list[INI_Line]
    is_header: bool = False

    def has_name(self, name: str) -> bool:
        """Check if the section has a specific name"""
        return self.name.strip().lower()[1:].strip("]") == name.strip().lower()

    def name_startswith(self, name: str) -> bool:
        """Check if the section name starts with a specific string"""
        if len(self.name) == 0:
            return False
        return self.name.strip().lower()[1:].startswith(name.strip().lower())

    def add_lines(self, lines: str) -> None:
        """Add lines to the section"""
        self.clear_empty_ending_lines()
        for line in lines.splitlines(keepends=True):
            self.add_single_line(line)
        # we sanitize last line to have no more or less than 1 empty line at the end
        self.clear_empty_ending_lines()
        self.add_single_line("\n\n")

    def add_single_line(self, line: str) -> None:
        key_value: list[str] = line.split("=")
        if len(key_value) == 2:
            key: str = key_value[0]
            value: str = key_value[1]
            self.lines.append(INI_Line(key=key, value=value, is_value_pair=True))
        else:
            self.lines.append(INI_Line(key=line, value="", is_value_pair=False))

    def clear_empty_ending_lines(self) -> None:
        """Remove empty lines at the end of the section"""
        while self.lines and self.lines[-1].key.strip() == "":
            self.lines.pop()

    def comment_out(self) -> None:
        """Comment out the section"""
        self.clear_empty_ending_lines()
        self.name = f";{self.name}"
        for line in self.lines:
            line.key = f";{line.key}"
        self.add_single_line("\n\n")


class INI_file:
    """Class to represent an ini file"""

    def __init__(self, body: str) -> None:
        self.sections: list[Section] = []
        if body != "":
            self.split_in_sections(body)

    def split_in_sections(self, content: str) -> None:
        """Split the content into sections based on [section] headers"""
        lines: list[str] = content.splitlines(keepends=True)
        curr_section: Section = Section(name="", lines=[], is_header=True)
        self.sections: list[Section] = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("[") and stripped_line.endswith("]"):
                if curr_section:
                    self.sections.append(curr_section)
                section_name: str = line
                curr_section = Section(name=section_name, lines=[])
                continue
            curr_section.add_single_line(line)
        if curr_section:
            self.sections.append(curr_section)

    def clean_up_indentation(self) -> None:
        """Clean up indentation in the ini file content"""
        for s in self.sections:
            s.name = s.name.lstrip()
            depth: int = 0
            for line in s.lines:
                if line._stripped_lower_key == "":
                    continue
                if line.key_startswith("if"):
                    depth += 1
                elif line.key_startswith("endif"):
                    depth -= 1
                if (
                    line.key_startswith("if")
                    or line.key_startswith("elif")
                    or line.key_startswith("else")
                ):
                    line.key = "\t" * (depth - 1) + line.key.lstrip()
                else:
                    line.key = "\t" * depth + line.key.lstrip()
                # line.key = str(depth) + line.key
            s.clear_empty_ending_lines()
            s.add_single_line("\n")

    def __str__(self) -> str:
        """Reconstruct the ini file from sections"""
        content: list[str] = []
        for section in self.sections:
            if not section.is_header:
                content.append(section.name)
            for line in section.lines:
                if line.is_value_pair:
                    content.append(f"{line.key}={line.value}")
                    continue
                content.append(line.key)
        return "".join(content)
