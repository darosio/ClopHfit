"""ChatGPT"""
import logging
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List

import pyparsing

LOGGER = logging.getLogger(__name__)


@dataclass
class Metadata:
    """Metadata for labelblocks. ? and whole file."""

    measurement_chamber_temperature: float = 0.0
    current_measurement: str | None = None  # or: ""
    data: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class MetadataReader:
    metadata: Metadata = Metadata()

    @staticmethod
    def _parse_measurement_chamber_temperature(
        # tokens: List[pyparsing.ParseResults], metadata: Metadata
        tokens: pyparsing.ParseResults,
        metadata: Metadata,
    ) -> None:
        value = tokens[0][1]
        # Single token: metadata.measurement_chamber_temperature = float(tokens.value)
        metadata.measurement_chamber_temperature = float(value)

    def _parse_meas(self, tokens: pyparsing.ParseResults, metadata: Metadata) -> None:
        key = tokens.value
        metadata.current_measurement = key
        if key not in metadata.data:
            metadata.data[key] = {}
            metadata.data[key]["metadata"] = {}
            metadata.data[key]["metadata"][
                "measurement_chamber_temperature"
            ] = metadata.measurement_chamber_temperature

    @staticmethod
    def _parse_other(tokens: pyparsing.ParseResults, metadata: Metadata) -> None:
        if metadata.current_measurement is not None:
            key = tokens.name
            value = tokens.value
            metadata.data[metadata.current_measurement]["metadata"][key] = value

    def _generate_parser(self, keyword_list: List[str]) -> pyparsing.ParserElement:
        """
        Generate a pyparsing parser for the specified keyword list.

        Parameters
        ----------
        keyword_list: List[str]
            A list of strings specifying the keywords to parse.

        Returns
        -------
            A pyparsing object that defines a parser for the keywords.
        """
        EOL = pyparsing.LineEnd().suppress()  # type: ignore # noqa: N806
        w = pyparsing.Word(pyparsing.alphanums + ".\u00B0%")  # . | deg | %
        parsers = []

        for keyword in keyword_list:
            if keyword == "Measurement chamber temperature":
                parsers.append(pyparsing.Keyword(keyword) + w("value") + EOL)
            elif keyword == "Meas":
                parsers.append(pyparsing.Keyword(keyword) + w("value") + EOL)
            else:
                parsers.append(
                    pyparsing.Keyword(keyword)("name")
                    + pyparsing.ZeroOrMore(pyparsing.White(" \t")).suppress()
                    + w("value")
                    + pyparsing.Optional(w)
                    + EOL
                )

        pr = pyparsing.MatchFirst(parsers)

        def parse_action(tokens: pyparsing.ParseResults) -> None:
            if tokens[0] == "Meas":
                self._parse_meas(tokens, self.metadata)
            elif tokens[0] == "Measurement chamber temperature":
                self._parse_measurement_chamber_temperature(tokens, self.metadata)
            else:
                self._parse_other(tokens, self.metadata)

        pr.setParseAction(parse_action)

        return pr
