using System.ComponentModel.DataAnnotations;

namespace MedicalAssistant.Shared.DTOs.PatientDTOs
{
    /// <summary>
    /// DTO used to create a new patient.
    /// </summary>
    public class CreatePatientDto
    {
        [Required(ErrorMessage = "Full name is required")]
        [MaxLength(100, ErrorMessage = "Full name must not exceed 100 characters")]
        public string FullName { get; set; } = default!;

        [Required(ErrorMessage = "Email is required")]
        [EmailAddress(ErrorMessage = "Invalid email format")]
        [MaxLength(150, ErrorMessage = "Email must not exceed 150 characters")]
        public string Email { get; set; } = default!;

        [Required(ErrorMessage = "Phone number is required")]
        [MaxLength(20, ErrorMessage = "Phone number must not exceed 20 characters")]
        [Phone(ErrorMessage = "Invalid phone number format")]
        public string PhoneNumber { get; set; } = default!;

        [Required(ErrorMessage = "Date of birth is required")]
        public DateTime DateOfBirth { get; set; }

        [Required(ErrorMessage = "Gender is required")]
        [MaxLength(10, ErrorMessage = "Gender must not exceed 10 characters")]
        public string Gender { get; set; } = default!;

        [MaxLength(300, ErrorMessage = "Address must not exceed 300 characters")]
        public string? Address { get; set; }

        public string? ImageUrl { get; set; }

        [MaxLength(5, ErrorMessage = "Blood type must not exceed 5 characters")]
        public string? BloodType { get; set; }

        public string? MedicalNotes { get; set; }
    }
}
