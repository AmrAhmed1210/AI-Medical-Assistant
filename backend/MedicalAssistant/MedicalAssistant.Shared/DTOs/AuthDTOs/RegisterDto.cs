using System.ComponentModel.DataAnnotations;

namespace MedicalAssistant.Shared.DTOs.AuthDTOs
{
    public class RegisterDto
    {
        [Required]
        public string Name { get; set; } = string.Empty;

        [Required]
        [EmailAddress]
        public string Email { get; set; } = string.Empty;

        [Required]
        [MinLength(6)]
        public string PasswordHash { get; set; } = string.Empty;

        public string Role { get; set; } = "Patient";

        public string Phone { get; set; } = string.Empty;
    }
}