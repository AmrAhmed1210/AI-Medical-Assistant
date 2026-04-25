namespace MedicalAssistant.Shared.DTOs.AuthDTOs
{
    public class AuthResponseDto
    {
        public string AccessToken { get; set; } = string.Empty;
        public string RefreshToken { get; set; } = string.Empty;
        public int ExpiresIn { get; set; } = 86400;
        public UserDto User { get; set; } = null!;
    }

    public class UserDto
    {
        public int Id { get; set; } = 0;
        public string FullName { get; set; } = string.Empty;
        public string Email { get; set; } = string.Empty;
        public string Role { get; set; } = string.Empty;
        public string? PhotoUrl { get; set; }
    }
}