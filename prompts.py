"""Tất cả prompt của hệ thống gom về một chỗ để dễ tinh chỉnh (quan trọng cho demo).

- CHAT_PROMPT   : hỏi đáp theo ngữ cảnh (tab Chat)
- QUIZ_PROMPT   : sinh câu hỏi trắc nghiệm (tab Quiz)
- SUMMARY_PROMPT: tóm tắt một phần / cả tài liệu (tab Tóm tắt)
- REDUCE_PROMPT : gộp các bản tóm tắt từng phần thành một bản tổng hợp
"""

CHAT_PROMPT = """Bạn là trợ lý hỏi đáp. Dùng các đoạn ngữ cảnh dưới đây để trả lời câu hỏi.
Nếu ngữ cảnh không có thông tin, hãy nói là bạn không biết, đừng bịa.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh: {context}

Câu hỏi: {question}
Trả lời:"""

QUIZ_PROMPT = """Bạn là giáo viên ra đề. Dựa HOÀN TOÀN vào nội dung tài liệu dưới đây, hãy tạo {n} câu hỏi trắc nghiệm tiếng Việt để giúp học viên tự ôn tập.

Yêu cầu:
- Mỗi câu hỏi có đúng 4 lựa chọn, chỉ 1 đáp án đúng.
- Câu hỏi và đáp án phải bám sát nội dung tài liệu, KHÔNG bịa thông tin ngoài tài liệu.
- Các lựa chọn sai phải hợp lý (gây nhiễu), không quá lố.
- Đa dạng độ khó và bao quát nhiều phần của tài liệu.

Chỉ trả về DUY NHẤT một mảng JSON, KHÔNG kèm giải thích, KHÔNG dùng markdown, KHÔNG có chữ nào trước/sau. Định dạng chính xác:
[
  {{
    "question": "Nội dung câu hỏi?",
    "options": ["Lựa chọn A", "Lựa chọn B", "Lựa chọn C", "Lựa chọn D"],
    "answer": 0
  }}
]
Trong đó "answer" là chỉ số (0-3) của đáp án đúng trong mảng "options".

Nội dung tài liệu:
{context}
"""

SUMMARY_PROMPT = """Bạn là trợ lý tóm tắt tài liệu học tập. Hãy tóm tắt nội dung dưới đây bằng tiếng Việt.

Yêu cầu:
- Nêu các ý chính dưới dạng gạch đầu dòng, ngắn gọn, dễ hiểu.
- Bám sát nội dung tài liệu, KHÔNG thêm thông tin ngoài tài liệu, KHÔNG bịa.
- Nếu tài liệu có nhiều phần/chủ đề, nhóm các ý theo từng phần.
- Bắt đầu bằng 1 câu nêu tài liệu nói về cái gì, rồi mới liệt kê ý chính.

Nội dung tài liệu:
{context}

Bản tóm tắt:"""

REDUCE_PROMPT = """Dưới đây là các bản tóm tắt của từng phần trong cùng một tài liệu.
Hãy gộp chúng thành MỘT bản tóm tắt tổng hợp bằng tiếng Việt, mạch lạc, không lặp ý, giữ lại mọi ý chính quan trọng. Trình bày dạng gạch đầu dòng.

Các bản tóm tắt từng phần:
{context}

Bản tóm tắt tổng hợp:"""
