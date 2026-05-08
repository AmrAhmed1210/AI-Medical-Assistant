using System.ComponentModel.DataAnnotations;

namespace MedicalAssistant.Shared.DTOs.AuthDTOs
{
    public class RegisterDto
    {
        [Required]
        public string FullName { get; set; } = string.Empty;

        [Required]
        [EmailAddress]
        public string Email { get; set; } = string.Empty;

        [Required]
        [MinLength(6)]
        public string Password { get; set; } = string.Empty;

        public string Role { get; set; } = "Patient";

        public string PhoneNumber { get; set; } = string.Empty;

        public DateTime? DateOfBirth { get; set; }
        public string? Gender { get; set; }
        public string? BloodType { get; set; }
        public decimal? Weight { get; set; }
        public decimal? Height { get; set; }
        public string? SmokingStatus { get; set; }
    }
}