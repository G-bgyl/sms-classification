from projFunction import sqlCsvTitle


sqlCsvTitle('SELECT id, raw_content, scene_code from sms_messages where id>1278 ',['id', 'raw_content', 'scene_code'])