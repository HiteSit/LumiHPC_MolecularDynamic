import paramiko
import os
from pathlib import Path
import time

class HPCConnection:
    """
    A class to handle SSH connections to HPC systems with support for
    password-protected SSH keys.
    """
    
    def __init__(
        self,
        hostname = "lumi.csc.fi",
        username = "fuscoric",
        key_path = "~/.ssh/id_ed25519",
        key_passphrase=None
        ):
        """
        Initialize the HPC connection.
        
        Args:
            hostname (str): The hostname or IP address of the HPC system
            username (str): Your username on the HPC system
            key_path (str): Path to your SSH private key file
            key_passphrase (str, optional): Passphrase for your encrypted SSH key
        """
        self.hostname = hostname
        self.username = username
        self.key_path = os.path.expanduser(key_path)  # Expand ~ to home directory
        self.key_passphrase = key_passphrase
        self.client = None
        self.sftp = None
        
    def connect(self):
        """
        Establish SSH connection to the HPC using the provided key and passphrase.
        
        Raises:
            paramiko.ssh_exception.PasswordRequiredException: If key is encrypted and no passphrase provided
            paramiko.ssh_exception.SSHException: If connection fails for other SSH-related reasons
            FileNotFoundError: If the key file doesn't exist
        """
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            # Load the key with the passphrase if provided
            if os.path.isfile(self.key_path):
                key = paramiko.Ed25519Key.from_private_key_file(
                    self.key_path,
                    password=self.key_passphrase
                )
                self.client.connect(
                    self.hostname, 
                    username=self.username, 
                    pkey=key
                )
            else:
                raise FileNotFoundError(f"SSH key not found: {self.key_path}")
                
            # Open SFTP connection for file transfers
            self.sftp = self.client.open_sftp()
            print(f"Connected to {self.hostname}")
            
        except paramiko.ssh_exception.PasswordRequiredException:
            raise Exception("Your private key is encrypted. Please provide the passphrase using the key_passphrase parameter.")
        except Exception as e:
            # Clean up if connection fails
            if self.client:
                self.client.close()
            raise e
        
    def disconnect(self):
        """Close SSH and SFTP connections."""
        if self.sftp:
            self.sftp.close()
        if self.client:
            self.client.close()
            self.client = None
            self.sftp = None
        print(f"Disconnected from {self.hostname}")
        
    def run_command(self, command, working_dir=None):
        """
        Run a command on the HPC.
        
        Args:
            command (str): The command to execute
            working_dir (str, optional): Directory to change to before executing
                                         the command
                                         
        Returns:
            tuple: (output, error, exit_status) - The command's stdout, stderr, and exit code
        """
        if not self.client:
            raise Exception("Not connected. Call connect() first.")
            
        if working_dir:
            command = f"cd {working_dir} && {command}"
            
        print(f"Running: {command}")
        stdin, stdout, stderr = self.client.exec_command(command)
        exit_status = stdout.channel.recv_exit_status()
        
        output = stdout.read().decode()
        error = stderr.read().decode()
        
        if exit_status != 0:
            print(f"Command failed with status {exit_status}")
            print(f"Error: {error}")
        
        return output, error, exit_status
        
    def upload_file(self, local_path, remote_path):
        """
        Upload a single file to the HPC.
        
        Args:
            local_path (str): Path to the local file to upload
            remote_path (str): Destination path on the HPC
        """
        if not self.sftp:
            raise Exception("Not connected. Call connect() first.")
            
        # Expand local path if it contains ~
        local_path = os.path.expanduser(local_path)
        
        # Create remote directory if it doesn't exist
        remote_dir = os.path.dirname(remote_path)
        if remote_dir:
            try:
                self.sftp.stat(remote_dir)
            except FileNotFoundError:
                self._mkdir_p(remote_dir)
                
        print(f"Uploading {local_path} to {remote_path}")
        self.sftp.put(local_path, remote_path)
    
    def _mkdir_p(self, remote_directory):
        """
        Create remote directory and parents if they don't exist
        (similar to mkdir -p).
        
        Args:
            remote_directory (str): Path to create on the remote system
        """
        dirs = [d for d in remote_directory.split('/') if d]
        current_dir = ''
        
        for d in dirs:
            current_dir += '/' + d
            try:
                self.sftp.stat(current_dir)
            except FileNotFoundError:
                print(f"Creating remote directory: {current_dir}")
                self.sftp.mkdir(current_dir)
    
    def download_file(self, remote_path, local_path):
        """
        Download a single file from the HPC.
        
        Args:
            remote_path (str): Path to the remote file to download
            local_path (str): Destination path on the local system
        """
        if not self.sftp:
            raise Exception("Not connected. Call connect() first.")
            
        # Expand local path if it contains ~
        local_path = os.path.expanduser(local_path)
        
        # Create local directory if it doesn't exist
        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
            
        print(f"Downloading {remote_path} to {local_path}")
        self.sftp.get(remote_path, local_path)
    
    def upload_directory(self, local_dir, remote_dir):
        """
        Upload an entire directory to the HPC.
        
        Args:
            local_dir (str): Path to the local directory to upload
            remote_dir (str): Destination path on the HPC
        """
        if not self.sftp:
            raise Exception("Not connected. Call connect() first.")
            
        # Expand local path if it contains ~
        local_dir = os.path.expanduser(local_dir)
        local_path = Path(local_dir)
        
        # Create remote directory if it doesn't exist
        try:
            self.sftp.stat(remote_dir)
        except FileNotFoundError:
            self._mkdir_p(remote_dir)
            
        # Upload all files and subdirectories
        for item in local_path.glob('**/*'):
            if item.is_file():
                relative_path = item.relative_to(local_path)
                remote_path = f"{remote_dir}/{relative_path}"
                
                # Create remote parent directories if needed
                remote_parent = os.path.dirname(remote_path)
                try:
                    self.sftp.stat(remote_parent)
                except FileNotFoundError:
                    self._mkdir_p(remote_parent)
                    
                print(f"Uploading {item} to {remote_path}")
                self.sftp.put(str(item), remote_path)
    
    def download_directory(self, remote_dir, local_dir):
        """
        Download an entire directory from the HPC.
        
        Args:
            remote_dir (str): Path to the remote directory to download
            local_dir (str): Destination path on the local system
        """
        if not self.sftp:
            raise Exception("Not connected. Call connect() first.")
            
        # Expand local path if it contains ~
        local_dir = os.path.expanduser(local_dir)
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        
        files = self._get_all_remote_files(remote_dir)
        for remote_path in files:
            relative_path = remote_path.replace(remote_dir, '', 1).lstrip('/')
            local_file_path = local_path / relative_path
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"Downloading {remote_path} to {local_file_path}")
            self.sftp.get(remote_path, str(local_file_path))
            
    def _get_all_remote_files(self, remote_dir):
        """
        Recursively get all files in a remote directory.
        
        Args:
            remote_dir (str): Remote directory path to scan
            
        Returns:
            list: List of full paths to all files in the directory and subdirectories
        """
        files = []
        try:
            for entry in self.sftp.listdir_attr(remote_dir):
                remote_path = f"{remote_dir}/{entry.filename}"
                if self._is_dir(entry):
                    files.extend(self._get_all_remote_files(remote_path))
                else:
                    files.append(remote_path)
        except FileNotFoundError:
            print(f"Remote directory not found: {remote_dir}")
        return files
        
    def _is_dir(self, entry):
        """
        Check if a remote entry is a directory.
        
        Args:
            entry: SFTP file attribute entry
            
        Returns:
            bool: True if the entry is a directory, False otherwise
        """
        return entry.st_mode & 0o4000 == 0o4000
        
    def submit_job(self, script_path, working_dir=None):
        """
        Submit a job to the HPC queue system (typically SLURM).
        
        Args:
            script_path (str): Path to the job script on the remote system
            working_dir (str, optional): Directory to run the sbatch command from
            
        Returns:
            str or None: Job ID if submission was successful, None otherwise
        """
        if working_dir:
            command = f"cd {working_dir} && sbatch {script_path}"
        else:
            command = f"sbatch {script_path}"
            
        output, error, status = self.run_command(command)
        
        if status == 0:
            # Typically sbatch outputs "Submitted batch job JOBID"
            job_id = output.strip().split()[-1]
            print(f"Job submitted with ID: {job_id}")
            return job_id
        else:
            print(f"Failed to submit job: {error}")
            return None
    
    def check_job_status(self, job_id):
        """
        Check the status of a submitted job.
        
        Args:
            job_id (str): Job ID to check
            
        Returns:
            str: Status of the job (PENDING, RUNNING, COMPLETED, FAILED, etc.)
        """
        command = f"squeue -j {job_id}"
        output, error, status = self.run_command(command)
        
        if "Invalid job id specified" in error:
            # Check if it completed successfully with sacct
            sacct_cmd = f"sacct -j {job_id} -o State -n"
            sacct_out, _, _ = self.run_command(sacct_cmd)
            
            if "COMPLETED" in sacct_out:
                return "COMPLETED"
            elif "FAILED" in sacct_out:
                return "FAILED"
            elif "CANCELLED" in sacct_out:
                return "CANCELLED"
            else:
                return "UNKNOWN"
            
        if status != 0:
            return "UNKNOWN"
            
        # Parse squeue output to determine status
        lines = output.strip().split('\n')
        if len(lines) < 2:  # No job data returned
            return "COMPLETED"
            
        # Parse status from squeue output
        job_info = lines[1].split()
        if len(job_info) >= 5:
            return job_info[4]  # Status column in squeue output
        
        return "UNKNOWN"