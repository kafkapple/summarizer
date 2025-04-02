from notion_client import Client
from typing import List, Dict, Optional, Any
from omegaconf import DictConfig
import os
import json
from datetime import datetime
import logging
import re
from notion_client.errors import APIResponseError

logger = logging.getLogger(__name__)

class NotionBase:
    def __init__(self, cfg: DictConfig, verbose=False, quiet=False):
        self.cfg = cfg
        self.client = Client(auth=cfg.api_keys['notion'])
        self.database_id = cfg.notion.database_id
        self.verbose = verbose
        self.quiet = quiet
        self.mapping = cfg.notion.mapping

    def change_id(self, id):
        self.database_id = id

    def _create_notion_properties(self, data: Dict) -> Dict:
        """Notion 속성 생성 (매핑 기반)"""
        properties = {}
        if not self.mapping:
            logger.error("Notion mapping is not defined in config.yaml!")
            return {}

        # 기본 데이터 준비
        internal_data = data.copy()
        
        # LLM 관련 정보 추가
        internal_data['gpt_model'] = self.cfg.llm.get('model')
        internal_data['output_language'] = self.cfg.output.get('language')
        
        # 요약 데이터 처리
        summary_dict = internal_data.get('summary', {})
        if isinstance(summary_dict, dict):
            internal_data['keywords'] = summary_dict.get('keywords_original', [])
            internal_data['summary'] = summary_dict.get('full_summary', '')
            internal_data['one_sentence_summary'] = summary_dict.get('one_sentence_summary', '')
            internal_data['core_summary'] = summary_dict.get('core_summary', '')
            internal_data['full_text'] = summary_dict.get('full_text', '')
            internal_data['summary_strategy'] = summary_dict.get('summary_strategy', 'default')

        # 매핑된 속성 생성
        for notion_name, map_info in self.mapping.items():
            internal_key = map_info.get('key')
            notion_type = map_info.get('type')
            
            if not internal_key or not notion_type:
                logger.warning(f"Incomplete mapping for Notion property '{notion_name}'")
                continue

            value = internal_data.get(internal_key)
            if value is None or value == '':
                continue

            try:
                formatted_value = self._format_notion_value(value, notion_type)
                if formatted_value:
                    properties[notion_name] = formatted_value
            except Exception as e:
                logger.error(f"Error formatting property '{notion_name}' (type: {notion_type}) with value '{value}': {e}")

        return properties

    def _format_notion_value(self, value: any, notion_type: str) -> Optional[Dict]:
        """Notion 속성 타입에 따른 값 포맷팅"""
        try:
            if notion_type == 'title':
                return {'title': [{'text': {'content': str(value)}}]}
            elif notion_type == 'rich_text':
                return {'rich_text': [{'text': {'content': str(value)[:2000]}}]}
            elif notion_type == 'number':
                if isinstance(value, (int, float)):
                    return {'number': int(value)}
                elif isinstance(value, str):
                    # 시간 형식 처리 (HH:MM:SS)
                    if ':' in value:
                        try:
                            h, m, s = map(int, value.split(':'))
                            return {'number': h * 3600 + m * 60 + s}
                        except:
                            pass
                    # 숫자 문자열 처리
                    if value.isdigit():
                        return {'number': int(value)}
                logger.warning(f"Invalid number value: {value}")
                return None
            elif notion_type == 'url':
                if isinstance(value, str) and value.startswith('http'):
                    return {'url': value}
                logger.warning(f"Invalid URL value: {value}")
                return None
            elif notion_type == 'date':
                if isinstance(value, str) and value:
                    try:
                        iso_date = datetime.fromisoformat(value.replace('Z', '+00:00')).isoformat()
                        return {'date': {'start': iso_date}}
                    except ValueError:
                        logger.warning(f"Invalid date format: {value}")
                        return None
            elif notion_type == 'select':
                if isinstance(value, str) and value:
                    return {'select': {'name': value}}
                elif isinstance(value, list) and value:
                    # 리스트의 첫 번째 값을 사용
                    return {'select': {'name': str(value[0])}}
                logger.warning(f"Invalid select value: {value}")
                return None
            elif notion_type == 'multi_select':
                unique_items = []
                if isinstance(value, list):
                    if all(isinstance(v, dict) and 'name' in v for v in value):
                        seen_names = set()
                        for item in value:
                            name = item.get('name')
                            if name and name not in seen_names:
                                unique_items.append(item)
                                seen_names.add(name)
                    elif all(isinstance(v, str) for v in value):
                        seen_names = set()
                        for item in value:
                            if item and item not in seen_names:
                                unique_items.append({'name': item})
                                seen_names.add(item)
                elif isinstance(value, str) and value:
                    unique_items = [{'name': value}]

                if unique_items:
                    return {'multi_select': unique_items}
                logger.warning(f"Invalid multi_select value: {value}")
                return None
            elif notion_type == 'checkbox':
                return {'checkbox': bool(value)}
            else:
                logger.warning(f"Unsupported Notion type: {notion_type}")
                return None

        except Exception as e:
            logger.error(f"Error formatting value '{value}' for type '{notion_type}': {e}")
            return None

    def rich_text_cutter(self, text: str) -> List[Dict[str, Any]]:
        """Helper function to cut text to 2000 chars and format as Notion rich_text."""
        if not isinstance(text, str):
            text = str(text) # Ensure text is string
        
        # Truncate if necessary
        truncated_text = text[:2000]
        
        # Return in Notion rich_text format
        if not truncated_text:
            return [] # Return empty list if text is empty
        else:
            return [{'text': {'content': truncated_text}}]

    def _save_summary_to_md(self, data: Dict, filepath: str):
        """요약을 마크다운 파일로 저장"""
        try:
            summary_data = data.get('summary', {})
            if not isinstance(summary_data, dict):
                logger.warning("Cannot save summary to MD: 'summary' key does not contain a dictionary.")
                return

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {data.get('title', 'Untitled')}\n\n")

                if data.get('url'):
                    f.write(f"**URL:** {data['url']}\n\n")

                one_sentence = summary_data.get('one_sentence_summary', '')
                if one_sentence:
                    f.write(f"**One Sentence Summary:** {one_sentence}\n\n")

                full_summary = summary_data.get('full_summary', '')
                if full_summary:
                    f.write("## Full Summary\n")
                    f.write(f"{full_summary}\n\n")

                keywords = summary_data.get('keywords', [])
                if keywords:
                    f.write("## Keywords\n")
                    keyword_terms = [kw.get('term', '') for kw in keywords if kw.get('term')]
                    f.write(f"- {', '.join(keyword_terms)}\n\n")

                chapters = summary_data.get('chapters', [])
                sections = summary_data.get('sections', [])

                if chapters:
                    f.write("## Chapters & Sections\n")
                    for chap in chapters:
                        chapter_number = chap.get('chapter_number', '')
                        default_title = f'Chapter {chapter_number}' if chapter_number else 'Chapter'
                        chapter_title = chap.get('chapter_title', default_title)
                        f.write(f"### {chapter_title}\n")
                        for sec_idx, sec in enumerate(chap.get('sections', [])):
                            f.write(f"#### {sec_idx + 1}. {sec.get('title', 'Section')}\n")
                            for point in sec.get('summary', []):
                                f.write(f"- {point}\n")
                            f.write("\n")
                        f.write("\n")
                elif sections:
                    f.write("## Sections\n")
                    for sec_idx, sec in enumerate(sections):
                        f.write(f"### {sec_idx + 1}. {sec.get('title', 'Section')}\n")
                        for point in sec.get('summary', []):
                            f.write(f"- {point}\n")
                        f.write("\n")

            if not self.quiet:
                logger.info(f"Detailed summary saved to: {filepath}")

        except Exception as e:
            logger.error(f"Error saving summary to Markdown file {filepath}: {e}")

    def save_to_notion(self, properties: Dict[str, Any], children: List[Dict[str, Any]] = None) -> bool:
        """
        Saves the given properties and children to the Notion database.
        Handles potential API errors gracefully.
        """
        page_title_for_logging = "N/A"
        try:
            # Attempt to get title for logging early
            # Safely access nested dictionary keys
            if isinstance(properties.get("Title"), dict) and \
               isinstance(properties["Title"].get("title"), list) and \
               properties["Title"]["title"] and \
               isinstance(properties["Title"]["title"][0], dict) and \
               isinstance(properties["Title"]["title"][0].get("text"), dict) and \
               properties["Title"]["title"][0]["text"].get("content"):
                page_title_for_logging = properties["Title"]["title"][0]["text"]["content"]

            # Debug logging before API call
            if self.verbose or logger.isEnabledFor(logging.DEBUG):
                try:
                    props_json = json.dumps(properties, ensure_ascii=False, indent=2)
                    children_json = json.dumps(children, ensure_ascii=False, indent=2) if children else "[]"
                    logger.debug(f"--- Sending to Notion (DB: {self.database_id}) ---")
                    logger.debug(f"Properties:\n{props_json}")
                    logger.debug(f"Children:\n{children_json}")
                    logger.debug("---------------------------------------")
                except Exception as dump_e:
                    logger.error(f"Could not dump properties/children to JSON for debugging: {dump_e}")

            # Notion API call
            response = self.client.pages.create(
                parent={"database_id": self.database_id},
                properties=properties,
                children=children if children else [] # Pass empty list if children is None
            )

            # Success logging
            if isinstance(response, dict) and response.get("object") == "page":
                page_id = response.get('id', 'N/A')
                page_url = response.get('url', 'N/A')
                logger.info(f"Successfully saved to Notion: '{page_title_for_logging}' (ID: {page_id}, URL: {page_url})")
                return True
            else:
                # Unexpected successful response format
                logger.warning(f"Unexpected response format after saving '{page_title_for_logging}' to Notion: {response}")
                return False

        except APIResponseError as e:
            # Handle specific Notion API errors
            error_body = e.body
            error_message = str(error_body)
            try:
                if isinstance(error_body, str):
                    error_details = json.loads(error_body)
                    error_message = error_details.get('message', str(error_details))
                elif isinstance(error_body, dict):
                    error_message = error_body.get('message', str(error_body))
            except json.JSONDecodeError:
                pass
            except Exception as parse_e:
                logger.error(f"Error parsing Notion API error body: {parse_e}")

            logger.error(f"Notion API Error saving page '{page_title_for_logging}': Code={e.code}, Message='{error_message}'")

            # Log failed properties/children
            try:
                failed_props_json = json.dumps(properties, ensure_ascii=False, indent=2)
                failed_children_json = json.dumps(children, ensure_ascii=False, indent=2) if children else "[]"
                logger.error(f"Failed properties causing API error:\n{failed_props_json}")
                logger.error(f"Failed children causing API error:\n{failed_children_json}")
            except Exception as dump_e:
                logger.error(f"Could not dump failed properties/children to JSON: {dump_e}")
            return False

        except Exception as e:
            # Handle other potential errors
            logger.error(f"Generic Error saving '{page_title_for_logging}' to Notion: {str(e)}", exc_info=True)
            # Log failed properties/children
            try:
                failed_props_json = json.dumps(properties, ensure_ascii=False, indent=2)
                failed_children_json = json.dumps(children, ensure_ascii=False, indent=2) if children else "[]"
                logger.error(f"Failed properties during generic error:\n{failed_props_json}")
                logger.error(f"Failed children during generic error:\n{failed_children_json}")
            except Exception as dump_e:
                logger.error(f"Could not dump failed properties/children to JSON: {dump_e}")
            return False

    def create_text_block(self, content: str, block_type: str = "paragraph", keywords: List[str] = None) -> Dict:
        """텍스트 블록 생성"""
        if keywords and self.cfg.get('highlight_keywords', False):
            rich_text = self.highlight_keywords(content, keywords)
        else:
            rich_text = [{"type": "text", "text": {"content": content}}]
        
        return {
            "object": "block",
            "type": block_type,
            block_type: {
                "rich_text": rich_text
            }
        }

    def create_bulleted_list_item(self, content: str, keywords: List[str] = None) -> Dict:
        """불릿 리스트 아이템 생성"""
        if keywords and self.cfg.get('highlight_keywords', False):
            rich_text = self.highlight_keywords(content, keywords)
        else:
            rich_text = [{"type": "text", "text": {"content": content}}]
        
        return {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": rich_text
            }
        }

    def organize_summary(self, data: Dict) -> List[Dict]:
        """요약 내용을 Notion 블록으로 구성 (JSON 출력과 유사하게)"""
        children = []
        max_blocks = 95 # API 제한 고려 (헤더 포함)

        try:
            # 순서 변경: 썸네일 -> 한 문장 -> 전체 -> 핵심 -> 상세 -> 키워드 -> 설명

            # 1. 썸네일 이미지 추가
            if data.get('thumbnail') and len(children) < max_blocks:
                children.append({
                    "object": "block", "type": "image",
                    "image": {"type": "external", "external": {"url": data['thumbnail']}}
                })
                if len(children) < max_blocks: children.append(self.create_text_block("")) # 구분선

            # 요약 정보 추출 (키워드 하이라이팅 준비)
            summary_dict = data.get('summary', {})
            if not isinstance(summary_dict, dict):
                logger.warning("'summary' key is not a dictionary in organize_summary.")
                summary_dict = {}

            highlight_keywords_terms = []
            if self.cfg.get('highlight_keywords', False):
                keywords_list = summary_dict.get('keywords', [])
                highlight_keywords_terms = [kw.get('term', '') for kw in keywords_list if isinstance(kw, dict) and kw.get('term')]

            # 2. 한 문장 요약
            one_sentence = summary_dict.get('one_sentence_summary')
            if one_sentence and len(children) < max_blocks:
                children.append(self.create_text_block("One Sentence Summary", "heading_2"))
                if len(children) < max_blocks: children.append(self.create_text_block(one_sentence, "paragraph", keywords=highlight_keywords_terms))
                if len(children) < max_blocks: children.append(self.create_text_block(""))

            # 3. 전체 요약 (Full Summary)
            full_summary = summary_dict.get('full_summary')
            if full_summary and len(children) < max_blocks:
                children.append(self.create_text_block("Full Summary", "heading_2"))
                # full_summary도 길 수 있으므로 나누기 (단순화 위해 여기서는 한 블록)
                # TODO: 2000자 이상일 경우 여러 블록으로 나누는 로직 추가 고려
                if len(children) < max_blocks: children.append(self.create_text_block(full_summary[:2000], "paragraph", keywords=highlight_keywords_terms))
                if len(children) < max_blocks: children.append(self.create_text_block(""))

            # 4. 핵심 요약 (Core Summary) - 구조 변경
            structured_core_summary = summary_dict.get('core_summary', []) # [{title:'..', points:['..',..]}, ..]

            if structured_core_summary and isinstance(structured_core_summary, list) and len(children) < max_blocks:
                children.append(self.create_text_block("Core Summary", "heading_2"))

                core_group_counter = 1
                for core_item in structured_core_summary:
                    if len(children) >= max_blocks: break
                    if isinstance(core_item, dict):
                        item_title = core_item.get('title', f'Topic {core_group_counter}')
                        item_points = core_item.get('points', [])

                        # 핵심 주제 제목 (넘버링 포함)
                        if len(children) < max_blocks: children.append(self.create_text_block(f"{core_group_counter}. {item_title}", "heading_3", keywords=highlight_keywords_terms))
                        core_group_counter += 1

                        # 핵심 주제별 bullet points
                        if isinstance(item_points, list):
                            for point in item_points:
                                if len(children) >= max_blocks: break
                                if isinstance(point, str) and point.strip():
                                    children.append(self.create_bulleted_list_item(point, keywords=highlight_keywords_terms))
                        elif isinstance(item_points, str) and item_points.strip(): # 혹시 문자열로 올 경우
                            if len(children) < max_blocks:
                                 children.append(self.create_bulleted_list_item(item_points, keywords=highlight_keywords_terms))

                        if len(children) < max_blocks: children.append(self.create_text_block("")) # 각 핵심 주제 그룹 후 공백

            # 5. 섹션별 상세 요약 (Detailed Summary Sections) - 넘버링 추가
            sections = summary_dict.get('sections', [])
            chapters = summary_dict.get('chapters', []) if self.cfg.get('enable_chapters', True) else []
            sections_summary_str = summary_dict.get('sections_summary', '') # sum.py에서 생성된 문자열

            has_detailed_content = (chapters and isinstance(chapters, list)) or \
                                   (sections and isinstance(sections, list)) or \
                                   (sections_summary_str and isinstance(sections_summary_str, str))

            if has_detailed_content and len(children) < max_blocks:
                children.append(self.create_text_block("Detailed Summary Sections", "heading_1")) # 이 부분은 넘버링 없이 유지

                detailed_group_counter = 1 # 상세 요약 내 그룹 넘버링용

                # 챕터/섹션 처리 (챕터 우선)
                if chapters and isinstance(chapters, list):
                    for chap_idx, chapter in enumerate(chapters):
                        if len(children) >= max_blocks: break
                        if isinstance(chapter, dict):
                            chap_title = chapter.get("chapter_title", f"Chapter {chap_idx+1}")
                            # 챕터 제목 (heading_2) 추가 - 넘버링 포함
                            if len(children) < max_blocks: children.append(self.create_text_block(f"{detailed_group_counter}. {chap_title}", "heading_2", keywords=highlight_keywords_terms))
                            detailed_group_counter += 1

                            chapter_sections = chapter.get('sections', [])
                            if isinstance(chapter_sections, list):
                                for sec_idx, segment in enumerate(chapter_sections):
                                    if len(children) >= max_blocks: break
                                    if isinstance(segment, dict):
                                        sec_title = segment.get("title", f"Section {sec_idx+1}")
                                        # 섹션 제목 (heading_3) 추가 - 여기는 넘버링 제외 (챕터 하위 항목이므로)
                                        if len(children) < max_blocks: children.append(self.create_text_block(sec_title, "heading_3", keywords=highlight_keywords_terms))

                                        summary_content = segment.get('summary', [])
                                        if isinstance(summary_content, list):
                                            for item in summary_content:
                                                if len(children) >= max_blocks: break
                                                if isinstance(item, str):
                                                     if len(children) < max_blocks: children.append(self.create_bulleted_list_item(item, keywords=highlight_keywords_terms))
                                        elif isinstance(summary_content, str) and summary_content.strip(): # 문자열도 처리
                                             if len(children) < max_blocks: children.append(self.create_text_block(summary_content[:2000], "paragraph", keywords=highlight_keywords_terms)) # 길이 제한 추가
                                        if len(children) < max_blocks: children.append(self.create_text_block("")) # 섹션 간 공백
                            elif chapter.get('summary'): # 챕터 요약이 문자열로 있는 경우
                                if len(children) < max_blocks: children.append(self.create_text_block(chapter['summary'][:2000], "paragraph", keywords=highlight_keywords_terms))
                                if len(children) < max_blocks: children.append(self.create_text_block(""))

                        if len(children) >= max_blocks: break # 챕터 루프 탈출

                # 챕터 없이 섹션만 처리
                elif sections and isinstance(sections, list):
                    for i, segment in enumerate(sections):
                        if len(children) >= max_blocks: break
                        if isinstance(segment, dict):
                            sec_title = segment.get("title", f"Section {i+1}")
                            # 섹션 제목 (heading_2) 추가 - 넘버링 포함
                            if len(children) < max_blocks: children.append(self.create_text_block(f"{detailed_group_counter}. {sec_title}", "heading_2", keywords=highlight_keywords_terms))
                            detailed_group_counter += 1

                            summary_content = segment.get('summary', [])
                            if isinstance(summary_content, list):
                                for item in summary_content:
                                    if len(children) >= max_blocks: break
                                    if isinstance(item, str):
                                        if len(children) < max_blocks: children.append(self.create_bulleted_list_item(item, keywords=highlight_keywords_terms))
                            elif isinstance(summary_content, str) and summary_content.strip(): # 문자열도 처리
                                if len(children) < max_blocks: children.append(self.create_text_block(summary_content[:2000], "paragraph", keywords=highlight_keywords_terms)) # 길이 제한 추가
                            if len(children) < max_blocks: children.append(self.create_text_block("")) # 섹션 간 공백
                        if len(children) >= max_blocks: break # 섹션 루프 탈출

                # 문자열 형태의 sections_summary만 있을 때
                elif sections_summary_str and isinstance(sections_summary_str, str):
                    # 이 경우 넘버링 없이 전체 내용을 표시
                    if len(children) < max_blocks: children.append(self.create_text_block(sections_summary_str[:2000], "paragraph", keywords=highlight_keywords_terms)) # 길이 제한 추가
                    if len(children) < max_blocks: children.append(self.create_text_block(""))

                # Detailed Summary 섹션 전체 후 공백 (이미 섹션 간 공백이 있어서 불필요할 수 있음)
                # if len(children) < max_blocks: children.append(self.create_text_block(""))


            # 6. 키워드
            keywords_list = summary_dict.get('keywords', [])
            if keywords_list and len(children) < max_blocks:
                children.append(self.create_text_block("Keywords", "heading_2"))
                kw_strings = []
                for kw in keywords_list:
                    if isinstance(kw, dict):
                        term = kw.get('term', 'N/A')
                        freq = kw.get('frequency')
                        kw_strings.append(f"{term}{f' ({freq})' if freq else ''}")
                    elif isinstance(kw, str):
                        kw_strings.append(kw)
                if kw_strings and len(children) < max_blocks:
                    # 키워드를 하나의 불릿 포인트로
                    children.append(self.create_bulleted_list_item(", ".join(kw_strings)))
                if len(children) < max_blocks: children.append(self.create_text_block(""))


            # 7. 설명 (Description) - 마지막으로 이동
            description = data.get('description')
            if description and len(children) < max_blocks:
                children.append(self.create_text_block("Description", "heading_2"))
                if len(children) < max_blocks: children.append(self.create_text_block(description[:2000], "paragraph")) # 길이 제한
                if len(children) < max_blocks: children.append(self.create_text_block("")) # 구분선


            # 8. 요약 전략 (원래 위치 유지 또는 이동 고려 - 여기서는 순서 변경 요청에 없으므로 일단 유지)
            summary_strategy = summary_dict.get('summary_strategy_used', data.get('summary_strategy'))
            if summary_strategy and len(children) < max_blocks:
                children.append(self.create_text_block("Summary Strategy Used", "heading_2"))
                if len(children) < max_blocks: children.append(self.create_text_block(summary_strategy, "paragraph"))
                # if len(children) < max_blocks: children.append(self.create_text_block("")) # 마지막 항목 뒤 공백 제거

        except Exception as e:
            logger.error(f'Error organizing summary content: {e}', exc_info=True)

        # 최종 블록 수 반환 (API 제한 내에서)
        return children

    def highlight_keywords(self, text: str, keywords: List[str]) -> List[Dict]:
        """키워드 하이라이팅"""
        if not text or not keywords:
            return [{"type": "text", "text": {"content": text or ""}}]

        result = []
        current_pos = 0
        text_lower = text.lower()
        
        sorted_keywords = sorted(keywords, key=len, reverse=True)
        
        while current_pos < len(text):
            found_keyword = False
            
            for keyword in sorted_keywords:
                keyword_lower = keyword.lower()
                pos = text_lower.find(keyword_lower, current_pos)
                
                if pos == current_pos:
                    if pos > 0:
                        result.append({
                            "type": "text",
                            "text": {"content": text[current_pos:pos]}
                        })
                    
                    result.append({
                        "type": "text",
                        "text": {
                            "content": text[pos:pos + len(keyword)],
                        },
                        "annotations": {
                            "bold": True,
                            "color": "yellow_background"
                        }
                    })
                    
                    current_pos = pos + len(keyword)
                    found_keyword = True
                    break
            
            if not found_keyword:
                next_pos = len(text)
                for keyword in sorted_keywords:
                    keyword_pos = text_lower.find(keyword_lower, current_pos + 1)
                    if keyword_pos != -1 and keyword_pos < next_pos:
                        next_pos = keyword_pos
                
                result.append({
                    "type": "text",
                    "text": {"content": text[current_pos:next_pos]}
                })
                current_pos = next_pos
        
        return result



