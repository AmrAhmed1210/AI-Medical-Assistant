namespace backend.DTOs
{
    public class UserDto
    {
        public string Name { get; set; } = string.Empty;
        public string Email { get; set; } = string.Empty;
        public string PasswordHash { get; set; } = string.Empty; // ده اللي الموبايل بيبعته ككلمة سر عادية والسيرفر بيشفرها
        public string? Role { get; set; } = "Patient";
    }
}