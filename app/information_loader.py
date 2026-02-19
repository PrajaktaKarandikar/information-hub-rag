""" 
Information loader module for handling document uploads and downloads from S3, web and local file system.
Also has backup mechanisms for loading documents in case of S3 issues. 
"""

import os
import traceback
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from bs4 import BeautifulSoup
from pypdf import PdfReader
import requests
import json
import tempfile
import hashlib
import validators
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

class InformationLoader:
    def __init__(self, use_s3: bool = True, cache_dir: str = "./cache"):
        self.use_s3 = use_s3
        self.cache_dir = cache_dir
        self.s3_client = None
        self.s3_bucket = os.getenv('AWS_S3_BUCKET_NAME', 'pj-rag-2026')
        self._s3_initialized = False 

        # Create a cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize S3 client if use_s3 is True
        if use_s3 and not self._s3_initialized: # Check flag to avoid re-initialization
            self._init_s3_client()
            self._s3_initialized = True # Set flag to indicate S3 client has been initialized

    def _init_s3_client(self):
        """ 
        Initializes the S3 client using credentials from environment variables. 
        Uses backup if credentials are missing or invalid. 
        """
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

        # import traceback  # check traces to see what is calling this method
        # print("🔍 _init_s3_client called from:")
        # traceback.print_stack(limit=5)

        if not access_key or not secret_key:
            logger.warning("AWS credentials not found in environment variables. S3 functionality is disabled.")
            self.use_s3 = False
            return
        
        try:
            import botocore.config
            config = botocore.config.Config(
                connect_timeout=5,
                read_timeout=5,
                retries={'max_attempts': 1}
            )

            self.s3_client = boto3.client(
            's3',
            aws_access_key_id= access_key,
            aws_secret_access_key = secret_key,
            region_name = os.getenv('AWS_REGION', 'us-east-1'),
            config=config
            )
            self.s3_client.head_bucket(Bucket=self.s3_bucket)
            logger.info(f"S3 client initialized successfully. Bucket: {self.s3_bucket}")
        except Exception as e:
            logger.warning(f"Error initializing S3 client: {e}. Using backup mechanism.")
            self.use_s3 = False
            
    def load_information(self, source: str) ->  str:
        """
        Loads a content/information from S3, web or a local file.
        
        Args:
            source (str): The source of the document. 
                - Can be an S3 key (s3://bucket/key)
                - A URL (http:// or https:// prefix)
                - Local file path.
                - Direct text input (if it doesn't match the above patterns)
        
        Returns:
            str: Extracted text content. 
        """

        try:
            if source.startswith("s3://") and self.use_s3:
                # Load from S3
                return self._load_from_s3(source)
            elif validators.url(source):
                # Load from web
                return self._load_from_web(source)
            elif source.endswith(".pdf"):
                # Load from a pdf file
                return self._load_from_pdf(source)
            elif os.path.exists(source):
                # Load from local file system
                return self._load_from_local_file(source)
            else:
                # Treat as direct text input
                logger.info("Source does not match S3, URL or file patterns. Treating as direct text input.")
                return source
        except Exception as e:
            logger.error(f"Failed to load information from source: {source}. Error: {e}")
            raise

    def _load_from_s3(self, s3_uri: str) -> str:
        """
        Load content from S3.

        Args:
            s3_uri (str): The S3 URI of the document.

        Returns:
            str: The extracted text content.
        """

        if not self.use_s3 or not self.s3_client:
            raise ValueError("S3 client is not initialized or S3 is unavailable.") 
        
        # Parse S3 URI
        path = s3_uri[5:]  # Remove 's3://'
        parts = path.split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI format. {s3_uri} should be in the format s3://bucket/key")
        
        bucket, key = parts
        
        try:
            # Download the file to a temporary location
            with tempfile.NamedTemporaryFile(suffix='.tmp', delete=False) as tmp_file:
                self.s3_client.download_file(bucket, key, tmp_file.name)
                tmp_path = tmp_file.name

            # Load the content from temporary file based on its type
            if key.endswith('.pdf'):
                content = self._load_from_pdf(tmp_path)
            else:
                content = self._load_from_local_file(tmp_path)

            # Clean up the temporary file
            os.unlink(tmp_path)
            return content
        
        except ClientError as e:
            logger.error(f"Failed downloading content from S3: {e}")
            raise

    def _load_from_web(self, url: str) -> str:
        """
        Load content from a web URL.        

        Args:
            url (str): The web URL of the document.

        Returns:
            str: The extracted text content.
        """
        try:
            # Add a proper User-Agent header
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36  (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for tag in soup(["script", "style"]):
                tag.decompose()

            text = soup.get_text(separator='\n', strip=True)

            # Ensure the text is a string
            if not isinstance(text, str):
                text = str(text)

            # Limit the text to a reasonable length 
            if len(text) > 10000:
                text = text[:10000] + "... [truncated]"

            return text
        
        except requests.RequestException as e:
            logger.error(f"Failed to load content from web URL: {url}. Error: {e}")
            raise

    def _load_from_pdf(self, file_path: str) -> str:
        """
        Load  and extract text content from a PDF file.

        Args:
            file_path (str): The path to the PDF file.

        Returns:
            str: The extracted text content.
        """
        try:
            reader = PdfReader(file_path)
            texts = []
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    texts.append(text.strip())

            return "\n\n".join(texts)
        
        except Exception as e:
            logger.error(f"Failed to load content from PDF: {file_path}. Error: {e}")
            raise
    
    def _load_from_local_file(self, file_path: str) -> str:
        """
        Load text content from a local file.
        
        Args:
            file_path (str): The path to the local file.

        Returns:
            str: The extracted text content.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
            
        except Exception as e:
            logger.error(f"Failed to load content from local file: {file_path}. Error: {e}")
            raise

    # Keep the following methods for uploading and downloading documents to/from S3, which can be used for backup or manual operations.
    def upload_content(self, file_path: str, s3_key: str):
        """ 
        Uploads the content/document to the specified S3 bucket.

        Args:
            file_path (str): The local path of the file to be uploaded.
            s3_key (str): The key (path) to store the file in the S3 bucket.
        
        Returns:
            None
        """

        if not self.use_s3:
            logger.warning("S3 client is not initialized or S3 is unavailable for upload.")
            return False
        
        try:
            self.s3_client.upload_file(file_path, self.s3_bucket, s3_key)
            logger.info(f"{file_path} Document uploaded successfully to s3://{self.s3_bucket}/{s3_key}")
            return True
        
        except ClientError as e:
            logger.error(f"Failed to upload document to S3: {e}")
            return False
        
    def download_content(self, s3_key: str, local_path: str):
        """
        Download content from S3 to a local path.

        Args:
            s3_key (str): The key (path) of the file in the S3 bucket.
            local_path (str): The local path to save the downloaded file.

        Returns:
            None
        """

        if not self.use_s3:
            logger.warning("S3 client is not initialized or S3 is unavailable for download.")
            return False
        
        try:
            self.s3_client.download_file(self.s3_bucket, s3_key, local_path)
            logger.info(f"Document downloaded successfully from s3://{self.s3_bucket}/{s3_key} to {local_path}")
            return True
        
        except ClientError as e:
            logger.error(f"Failed to download document from S3: {e}")
            return False