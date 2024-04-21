import re

def pattern_match(text):
    email_pattern = r'\b([A-Za-z0-9._%+-]{2})([A-Za-z0-9._%+-]+)@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'(\d{3})-(\d{4})-(\d{4})'
    resident_id_pattern = r'(\d{6})-?([1-4])(\d{6})'

    # 이메일 마스킹
    def mask_email(match):
        username, domain = match.group().split('@')
        username = username[0]+'*'*len(username[1:])
        domain = '*'*len(domain.split('.')[0])+'.'+'*'*len(domain.split('.')[1])
        return f"{username}@{domain}"

    # 전화번호 마스킹
    def mask_phone(match):
        return match.group(1) + "-****-****"

    # 주민등록번호 마스킹
    def mask_resident_id(match):
        return match.group().split('-')[0] + '-'+ len(match.group().split('-')[1])*'*'
    
    # 이메일 마스킹 적용
    masked_text = re.sub(email_pattern, mask_email, text)

    # 전화번호 마스킹 적용
    masked_text = re.sub(phone_pattern, mask_phone, masked_text)

    # 주민등록번호 마스킹 적용
    masked_text = re.sub(resident_id_pattern, mask_resident_id, masked_text)

    return masked_text