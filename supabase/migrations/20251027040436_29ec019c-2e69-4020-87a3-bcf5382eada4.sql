-- Add file metadata columns to messages table
ALTER TABLE public.messages
ADD COLUMN file_name TEXT,
ADD COLUMN file_type TEXT,
ADD COLUMN file_size INTEGER;