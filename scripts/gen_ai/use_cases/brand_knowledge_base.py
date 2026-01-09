# TestPilot Brand Knowledge Base
# Comprehensive brand context for Bluehost and Network Solutions

"""
This module provides extensive brand knowledge to TestPilot for intelligent test automation.
It includes URL patterns, product offerings, navigation structures, form fields, and common workflows.
"""

BRAND_KNOWLEDGE_BASE = {
    "bluehost": {
        "display_name": "Bluehost",
        "domain": "www.bluehost.com",
        "variations": ["bhcom", "BHCOM", "bluehost", "bh", "BH"],

        "base_urls": {
            "production": "https://www.bluehost.com",
            "my_account": "https://www.bluehost.com/my-account",
            "cart": "https://www.bluehost.com/cart",
            "checkout": "https://www.bluehost.com/registration",
            "hosting": "https://www.bluehost.com/hosting",
            "wordpress": "https://www.bluehost.com/wordpress",
            "domains": "https://www.bluehost.com/domains",
            "support": "https://www.bluehost.com/help",
            "blog": "https://www.bluehost.com/blog",
            "knowledge_base": "https://www.bluehost.com/help",
            "pricing": "https://www.bluehost.com/pricing"
        },

        "navigation": {
            "main_menu": {
                "WordPress": {
                    "submenu": [
                        "WordPress Hosting",
                        "WooCommerce",
                        "WordPress Cloud Hosting",
                        "VPS Hosting",
                        "Dedicated Hosting",
                        "Yoast SEO Plugin",
                        "Professional Web Design",
                        "Professional Digital Marketing",
                        "Pro Design Live Support"
                    ],
                    "patterns": [
                        "//a[@id='header-navbar-dropdown-0']",
                        "//a[@data-element-label='WordPress']",
                        "//nav//a[contains(text(), 'WordPress')]"
                    ]
                },
                "Hosting": {
                    "submenu": [
                        "Web Hosting",
                        "WordPress Hosting",
                        "VPS Hosting",
                        "Dedicated Hosting",
                        "SSL Certificates",
                        "Website Security",
                        "Website Backup",
                        "Website SEO Checker",
                        "Google Workspace",
                        "Professional Email"
                    ],
                    "patterns": [
                        "//a[@data-element-label='Hosting']",
                        "//nav//a[contains(text(), 'Hosting')]"
                    ]
                },
                "Domains": {
                    "submenu": [
                        "Domain Name Search",
                        "AI Domain Name Generator",
                        "Premium Domain Names",
                        "Domain Transfer",
                        "Domain Privacy + Protection",
                        "Domain Forwarding",
                        "Domain Expiration Protection"
                    ],
                    "patterns": [
                        "//a[@data-element-label='Domains']",
                        "//nav//a[contains(text(), 'Domains')]"
                    ]
                },
                "Pro Services": {
                    "submenu": [
                        "Professional Web Design",
                        "Professional Digital Marketing",
                        "Pro Design Live Support"
                    ]
                },
                "Agencies": {
                    "submenu": [
                        "Agency Hosting",
                        "Agency Partner Program"
                    ]
                },
                "Pricing": {
                    "submenu": []
                }
            },

            "dropdown_selectors": {
                "primary": "//section[contains(@class, 'dropdown')]",
                "menu_items": "//section[contains(@class, 'dropdown')]//a",
                "nav_sections": "//nav//section//a",
                "header_sections": "//header//section//a"
            }
        },

        "products": {
            "wordpress_hosting": {
                "plans": ["Basic", "Choice Plus", "Online Store", "Pro"],
                "page_url": "/wordpress/wordpress-hosting",
                "plan_selectors": {
                    "Basic": ["wpBasic", "WordPress Basic"],
                    "Choice Plus": ["wpChoicePlus", "WordPress Choice Plus"],
                    "Online Store": ["wpOnlineStore", "eCommerce"],
                    "Pro": ["wpPro", "WordPress Pro"]
                },
                "common_buttons": [
                    "View Plans",
                    "Choose Plan",
                    "Get Started",
                    "Select Plan",
                    "Add to Cart"
                ]
            },

            "web_hosting": {
                "plans": ["Basic", "Plus", "Choice Plus", "Pro"],
                "page_url": "/hosting/shared",
                "features": [
                    "Free Domain",
                    "Free SSL",
                    "Free CDN",
                    "Unmetered Bandwidth",
                    "WordPress Install"
                ]
            },

            "vps_hosting": {
                "plans": ["Standard", "Enhanced", "Ultimate"],
                "page_url": "/hosting/vps",
                "plan_names": {
                    "Standard": "Standard VPS - NVMe 4 with cPanel",
                    "Enhanced": "Enhanced VPS - NVMe 8 with cPanel",
                    "Ultimate": "Ultimate VPS - NVMe 16 with cPanel"
                }
            },

            "dedicated_hosting": {
                "plans": ["Standard", "Enhanced", "Premium"],
                "page_url": "/hosting/dedicated",
                "plan_names": {
                    "Standard": "Standard Dedicated - NVMe 32 with cPanel",
                    "Enhanced": "Enhanced Dedicated - NVMe 64 with cPanel",
                    "Premium": "Premium Dedicated - NVMe128 with cPanel"
                }
            },

            "cloud_hosting": {
                "plans": ["Cloud1", "Cloud10", "Cloud25", "Cloud50"],
                "page_url": "/cloud-hosting",
                "urls": {
                    "Cloud1": "/registration/?flow=jhfeCloud1",
                    "Cloud10": "/registration/?flow=jhfeCloud10",
                    "Cloud25": "/registration/?flow=jhfeCloud25",
                    "Cloud50": "/registration/?flow=jhfeCloud50"
                }
            },

            "woocommerce": {
                "plans": ["eCommerce Essentials", "eCommerce Premium"],
                "page_url": "/woocommerce-hosting",
                "features": [
                    "WooCommerce Pre-installed",
                    "Free SSL",
                    "Payment Gateways",
                    "Product Import/Export"
                ]
            },

            "domains": {
                "search_url": "/domains",
                "tlds": [".com", ".net", ".org", ".co", ".io", ".tech", ".blog", ".website", ".me"],
                "features": [
                    "Domain Privacy + Protection",
                    "Domain Expiration Protection",
                    "Domain Forwarding",
                    "Domain Transfer"
                ]
            },

            "email": {
                "google_workspace": {
                    "plans": ["Business Starter", "Business Standard", "Business Plus"],
                    "page_url": "/google-workspace",
                    "price_starter": "$3.00/month"
                },
                "professional_email": {
                    "features": ["Domain-branded email", "Webmail access", "POP/IMAP support"],
                    "page_url": "/email/professional-email",
                    "trial": "3-Month Free Trial"
                }
            }
        },

        "checkout_flow": {
            "steps": [
                "Domain Selection",
                "Package Selection",
                "Account Creation",
                "Payment Information",
                "Order Review",
                "Confirmation"
            ],

            "form_fields": {
                "account_creation": [
                    "email",
                    "password",
                    "firstName",
                    "lastName",
                    "phone"
                ],
                "billing_info": [
                    "address1",
                    "address2",
                    "city",
                    "state",
                    "zipcode",
                    "country"
                ],
                "payment": [
                    "cardNumber",
                    "expiryMonth",
                    "expiryYear",
                    "cvv",
                    "cardholderName"
                ]
            },

            "payment_methods": [
                "Credit Card",
                "PayPal",
                "Venmo",
                "Apple Pay",
                "Google Pay"
            ],

            "common_buttons": {
                "submit_payment": [
                    "Submit Payment",
                    "Complete Purchase",
                    "Place Order",
                    "Pay Now"
                ],
                "continue": [
                    "Continue",
                    "Next",
                    "Proceed",
                    "Continue to Payment"
                ]
            },

            "address_autocomplete": {
                "provider": "Google Places",
                "dropdown_wait": 10,
                "selection_required": True,
                "common_issues": [
                    "Dropdown appears late",
                    "Selection not registered",
                    "Manual entry after autocomplete"
                ]
            }
        },

        "common_patterns": {
            "cta_keywords": [
                "Buy", "Purchase", "Get Started", "Choose Plan", "Select",
                "Add to Cart", "View Plans", "Sign Up", "Register",
                "Continue", "Proceed", "Next", "Submit", "Complete"
            ],

            "navigation_keywords": [
                "Home", "Dashboard", "My Account", "Domains", "Hosting",
                "Email", "Support", "Billing", "Renew"
            ],

            "form_labels": {
                "email": ["Email", "Email Address", "Your Email"],
                "password": ["Password", "Create Password", "Your Password"],
                "phone": ["Phone", "Phone Number", "Contact Number"],
                "domain": ["Domain", "Domain Name", "Your Domain"]
            }
        },

        "test_data": {
            "sample_domains": [
                "test-automation-{timestamp}.com",
                "qa-test-{random}.com",
                "bh-test-{date}.com"
            ],
            "payment_test_card": {
                "number": "4532123456789012",
                "cvv": "123",
                "expiry_month": "12",
                "expiry_year": "2030"
            }
        },

        "known_issues": {
            "google_places_dropdown": {
                "issue": "Dropdown selection not completed",
                "wait_time": 10,
                "retry_attempts": 5
            },
            "submit_payment_button": {
                "issue": "Button not clicked accurately",
                "click_methods": ["standard", "actionchains", "javascript", "force"],
                "scroll_offset": -100
            },
            "late_dropdown": {
                "issue": "Dropdown appears after other fields filled",
                "solution": "Wait for dropdown before proceeding"
            }
        }
    },

    "network_solutions": {
        "display_name": "Network Solutions",
        "domain": "www.networksolutions.com",
        "variations": ["ncom", "NCOM", "nsol", "NSOL", "networksolutions", "NetworkSolutions", "network_solutions", "NetSol"],

        "base_urls": {
            "production": "https://www.networksolutions.com",
            "my_account": "https://www.networksolutions.com/my-account",
            "cart": "https://www.networksolutions.com/checkout",
            "checkout": "https://www.networksolutions.com/checkout/cart/create-your-account",
            "domains": "https://www.networksolutions.com/products/domain/domain-search-results",
            "hosting": "https://www.networksolutions.com/hosting",
            "wordpress": "https://www.networksolutions.com/wordpress-hosting",
            "email": "https://www.networksolutions.com/email",
            "ssl": "https://www.networksolutions.com/security/ssl-certificates",
            "website_builder": "https://www.networksolutions.com/website-builder",
            "support": "https://www.networksolutions.com/support",
            "domain_transfer": "https://www.networksolutions.com/domain-transfer",
            "ai_domain_generator": "https://www.networksolutions.com/domains/ai-domain-name-generator",
            "whois": "https://www.networksolutions.com/domains/whois",
            "domain_forwarding": "https://www.networksolutions.com/domains/domain-forwarding",
            "domain_privacy": "https://www.networksolutions.com/domains/private-domain-registration",
            "domain_expiration_protection": "https://www.networksolutions.com/domain-name-registration/domain-protect",
            "premium_domains": "https://www.networksolutions.com/domain-name-registration/domain-taken-center"
        },

        "navigation": {
            "main_menu": {
                "Domains": {
                    "submenu": [
                        "Domain Name Search",
                        "Domain Transfer",
                        "Premium Domain Names",
                        "Domain Privacy + Protection",
                        "Domain Expiration Protection",
                        "Domain Forwarding",
                        "WHOIS Search",
                        "Trademark Protection",
                        "AI Domain Name Generator",
                        "Domain Management"
                    ]
                },
                "Website & eCommerce": {
                    "submenu": [
                        "Website Builder",
                        "eCommerce Website Builder",
                        "Website Design Services",
                        "Online Marketing Services"
                    ]
                },
                "Hosting": {
                    "submenu": [
                        "Web Hosting",
                        "WordPress Hosting",
                        "VPS Hosting",
                        "Dedicated Hosting",
                        "Cloud Website Backup"
                    ]
                },
                "Security": {
                    "submenu": [
                        "SSL Certificates",
                        "SiteLock",
                        "CodeGuard Backup"
                    ]
                },
                "Email & Productivity": {
                    "submenu": [
                        "Professional Email",
                        "Google Workspace",
                        "Microsoft 365"
                    ]
                },
                "Professional Services": {
                    "submenu": [
                        "Logo Design",
                        "Web Design",
                        "SEO Services",
                        "PPC Advertising"
                    ]
                }
            }
        },

        "products": {
            "domains": {
                "search_url": "/products/domain/domain-search-results",
                "tlds": [".com", ".net", ".org", ".info", ".biz", ".us", ".co", ".io", ".online"],
                "features": {
                    "privacy_protection": "Domain Privacy + Protection",
                    "expiration_protection": "Domain Expiration Protection (DEP)",
                    "forwarding": "Domain Forwarding",
                    "dns_management": "DNS Management"
                },
                "transfer": {
                    "url": "/domain-transfer",
                    "requirements": ["Authorization code", "Unlock domain", "Over 60 days old"]
                }
            },

            "web_hosting": {
                "plans": ["Starter", "Professional", "Premium"],
                "page_url": "/hosting/web-hosting",
                "features": [
                    "Free Domain",
                    "Free SSL",
                    "Unlimited Bandwidth",
                    "24/7 Support"
                ]
            },

            "wordpress_hosting": {
                "plans": ["Starter", "Professional", "Premium"],
                "page_url": "/wordpress-hosting",
                "features": [
                    "Managed WordPress",
                    "Auto Updates",
                    "Daily Backups",
                    "WordPress Pre-installed"
                ]
            },

            "email": {
                "google_workspace": {
                    "plans": ["Business Starter", "Business Standard", "Business Plus"],
                    "page_url": "/email/google-workspace",
                    "mailbox_range": "1-300 users"
                },
                "professional_email": {
                    "page_url": "/email/professional-email",
                    "features": ["POP/IMAP", "Webmail", "Spam Protection"],
                    "mailbox_range": "1-300 users"
                }
            },

            "ssl": {
                "types": ["Xpress SSL", "Basic SSL", "Advanced SSL", "Wildcard SSL", "Extended Validation SSL"],
                "page_url": "/security/ssl-certificates",
                "validation_methods": ["HTTP", "CNAME", "Email"]
            },

            "website_builder": {
                "plans": ["Personal", "Business", "E-Commerce"],
                "page_url": "/website-builder",
                "features": [
                    "Drag & Drop Editor",
                    "Mobile Responsive",
                    "SEO Tools",
                    "Templates"
                ]
            },

            "ecommerce_website_builder": {
                "plans": ["Starter", "Professional", "Premium"],
                "page_url": "/ecommerce-website-builder",
                "features": [
                    "Shopping Cart",
                    "Payment Processing",
                    "Inventory Management",
                    "Product Catalog"
                ]
            }
        },

        "checkout_flow": {
            "steps": [
                "Product Selection",
                "Domain Selection/Entry",
                "Cart Review",
                "Account Creation",
                "Payment Information",
                "Order Confirmation"
            ],

            "form_fields": {
                "account_creation": [
                    "email",
                    "password",
                    "confirmPassword",
                    "firstName",
                    "lastName",
                    "phone",
                    "company",
                    "address",
                    "city",
                    "state",
                    "zip"
                ],
                "domain_contact": [
                    "firstName",
                    "lastName",
                    "organization",
                    "email",
                    "phone",
                    "address1",
                    "city",
                    "state",
                    "postalCode",
                    "country"
                ],
                "payment": [
                    "cardNumber",
                    "expirationMonth",
                    "expirationYear",
                    "securityCode",
                    "nameOnCard"
                ]
            },

            "payment_methods": [
                "Credit Card",
                "PayPal"
            ],

            "common_buttons": {
                "add_to_cart": [
                    "Add to Cart",
                    "Buy Now",
                    "Get Started"
                ],
                "continue": [
                    "Continue",
                    "Next",
                    "Proceed to Checkout"
                ],
                "complete_purchase": [
                    "Complete Purchase",
                    "Place Order",
                    "Submit Order"
                ]
            }
        },

        "common_patterns": {
            "cta_keywords": [
                "Search", "Transfer", "Buy", "Add to Cart", "Get Started",
                "Continue", "Register", "Sign Up", "Submit", "Complete Purchase"
            ],

            "navigation_keywords": [
                "Home", "My Account", "Dashboard", "Domains", "Hosting",
                "Email", "Support", "Cart", "Checkout"
            ],

            "form_labels": {
                "email": ["Email Address", "Email", "Your Email"],
                "domain": ["Domain Name", "Domain", "Enter Domain"],
                "mailboxes": ["Number of Users", "Mailboxes", "User Count"]
            }
        },

        "test_data": {
            "sample_domains": [
                "test-engg-ncom-automation{timestamp}.com",
                "te-ncom-qa-{date}.com",
                "ncom-test-{random}.com"
            ],
            "existing_domains": [
                "test-engg-ncom-automation123.com"
            ],
            "payment_test_card": {
                "number": "4111111111111111",
                "cvv": "123",
                "expiry_month": "12",
                "expiry_year": "2030"
            }
        },

        "validation_messages": {
            "invalid_domain": "Please enter a valid domain name.",
            "domain_required": "Please enter a domain name.",
            "domain_not_supported": "Sorry, this domain extension is not supported",
            "invalid_mailbox_quantity": "Invalid Entry: Please write a number between 1 and 300",
            "field_required": "This field is required"
        },

        "special_features": {
            "domain_plugin": {
                "max_length": 60,
                "with_spaces": "Generates domain suggestions with hyphens",
                "without_spaces": "Generates domain as-is"
            },
            "upsells": {
                "ssl_post_cart": "/upsell/ssl-post-cart",
                "dep_post_cart": "/upsell/dep-post-cart",
                "common": ["SSL", "DEP", "Privacy Protection", "Website Backup"]
            },
            "migrated_users": {
                "from_brands": ["Web.com (WCOM)", "Register.com (RCOM)", "Domain.com (DCOM)"],
                "authentication": "Can login with old credentials"
            }
        }
    }
}

# Brand-specific AI prompts and context
BRAND_AI_CONTEXT = {
    "bluehost": {
        "brand_description": """
        Bluehost is a leading web hosting provider recommended by WordPress.org since 2005.
        Known for reliable WordPress hosting, user-friendly interface, and 24/7 support.
        Target audience: Small businesses, bloggers, WordPress users, and agencies.
        Key differentiators: Official WordPress recommendation, 1-click WordPress install,
        free domain for 1 year, free SSL certificate, unmetered bandwidth.
        """,

        "user_experience_focus": [
            "Simplified WordPress setup process",
            "Clear pricing with no hidden fees",
            "Prominent CTAs for plan selection",
            "Easy-to-navigate control panel (cPanel/Bluehost custom panel)",
            "Mobile-responsive checkout flow",
            "Multiple payment options including digital wallets"
        ],

        "common_user_journeys": [
            "New user → Domain search → WordPress hosting plan → Account creation → Payment",
            "Existing user → Dashboard → Upgrade hosting → Payment",
            "Domain transfer → Select plan → DNS configuration → Complete transfer",
            "Email setup → Google Workspace → Configure mailboxes → Complete setup"
        ],

        "automation_tips": [
            "Wait for dropdown menus after hover (1s + detection)",
            "Use dropdown-specific selectors after navigation hover",
            "Google Places address autocomplete requires explicit selection",
            "Submit Payment button may need multiple click methods",
            "Check for Cloudflare protection before interacting"
        ]
    },

    "network_solutions": {
        "brand_description": """
        Network Solutions is a pioneer in domain registration (since 1991) and comprehensive
        web services provider. Known for premium domains, professional services, and enterprise solutions.
        Target audience: Businesses, professionals, domain investors, and enterprises.
        Key differentiators: 30+ years of domain experience, premium domain marketplace,
        professional design services, comprehensive security options, trademark protection.
        """,

        "user_experience_focus": [
            "Domain-first approach (domain search prominent)",
            "Professional/business-oriented design",
            "Detailed product information and comparisons",
            "Strong focus on security (SSL, privacy, protection)",
            "Multiple contact/support options",
            "Upsell opportunities throughout flow"
        ],

        "common_user_journeys": [
            "Domain search → Premium domain selection → Purchase",
            "Domain transfer → Authorization code → DNS transfer → Completion",
            "Hosting → Domain selection → Email add-on → Account creation → Payment",
            "SSL Certificate → Domain selection → Validation method → Installation",
            "Website Builder → Template selection → Domain → Launch"
        ],

        "automation_tips": [
            "Handle post-cart upsells (SSL, DEP commonly appear)",
            "Domain transfer requires authorization code input",
            "Email products ask for mailbox quantity (1-300 range validation)",
            "Multiple validation messages specific to Network Solutions",
            "Premium domain offers have special flow"
        ]
    }
}

# Intelligent brand detection patterns
BRAND_DETECTION_PATTERNS = {
    "url_patterns": {
        "bluehost": [
            r"bluehost\.com",
            r"registration\?.*flow=",
            r"/wordpress/",
            r"/hosting/shared"
        ],
        "network_solutions": [
            r"networksolutions\.com",
            r"/checkout/",
            r"/purchase-it/",
            r"actionController\.do"
        ]
    },

    "dom_patterns": {
        "bluehost": [
            "Bluehost",
            "recommended by WordPress",
            "cPanel",
            "WooCommerce"
        ],
        "network_solutions": [
            "Network Solutions",
            "Since 1991",
            "Domain Name Registration",
            "WHOIS"
        ]
    }
}

def get_brand_knowledge(brand_name: str) -> dict:
    """
    Get comprehensive brand knowledge for intelligent test automation

    Args:
        brand_name: Brand identifier (e.g., 'bluehost', 'bhcom', 'ncom', 'network_solutions')

    Returns:
        Dictionary containing comprehensive brand knowledge
    """
    # Normalize brand name
    brand_name_lower = brand_name.lower()

    # Check variations
    for brand_key, brand_data in BRAND_KNOWLEDGE_BASE.items():
        if brand_name_lower in [v.lower() for v in brand_data["variations"]]:
            return {
                **brand_data,
                "ai_context": BRAND_AI_CONTEXT.get(brand_key, {})
            }
        if brand_name_lower == brand_key:
            return {
                **brand_data,
                "ai_context": BRAND_AI_CONTEXT.get(brand_key, {})
            }

    # Return empty dict if brand not found
    return {}

def detect_brand_from_url(url: str) -> str:
    """
    Detect brand from URL using pattern matching

    Args:
        url: The URL to analyze

    Returns:
        Brand code for folder structure ('bhcom' for Bluehost, 'ncom' for Network Solutions, or 'unknown')
    """
    import re

    url_lower = url.lower()

    for brand, patterns in BRAND_DETECTION_PATTERNS["url_patterns"].items():
        for pattern in patterns:
            if re.search(pattern, url_lower):
                # Return the proper folder code instead of generic brand name
                if brand == "bluehost":
                    return "bhcom"
                elif brand == "network_solutions":
                    return "ncom"
                return brand

    return "unknown"

def get_brand_specific_selector(brand: str, element_type: str, context: str = "") -> list:
    """
    Get brand-specific selectors for common elements

    Args:
        brand: Brand name
        element_type: Type of element (e.g., 'submit_button', 'domain_input')
        context: Additional context (e.g., 'checkout', 'navigation')

    Returns:
        List of XPath/CSS selectors to try
    """
    brand_knowledge = get_brand_knowledge(brand)

    if not brand_knowledge:
        return []

    selectors = []

    # Add brand-specific patterns based on element type
    if element_type == "submit_button" and context == "checkout":
        if brand == "bluehost":
            button_texts = brand_knowledge.get("checkout_flow", {}).get("common_buttons", {}).get("submit_payment", [])
            for text in button_texts:
                selectors.extend([
                    f"//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text.lower()}')]",
                    f"//input[@type='submit' and contains(translate(@value, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text.lower()}')]"
                ])

    elif element_type == "navigation_menu":
        if "dropdown_selectors" in brand_knowledge.get("navigation", {}):
            selectors.extend(brand_knowledge["navigation"]["dropdown_selectors"].values())

    return selectors

def get_brand_ai_prompt_enhancement(brand: str, action: str) -> str:
    """
    Get brand-specific AI prompt enhancement for better context

    Args:
        brand: Brand name
        action: Action being performed (e.g., 'checkout', 'navigation', 'form_fill')

    Returns:
        Additional prompt context specific to the brand
    """
    brand_knowledge = get_brand_knowledge(brand)

    if not brand_knowledge or "ai_context" not in brand_knowledge:
        return ""

    ai_context = brand_knowledge["ai_context"]

    prompt_enhancement = f"""
Brand Context: {brand_knowledge.get('display_name', brand)}
{ai_context.get('brand_description', '')}

Key Considerations:
- User Experience Focus: {', '.join(ai_context.get('user_experience_focus', [])[:3])}
- Automation Tips: {', '.join(ai_context.get('automation_tips', [])[:2])}
"""

    return prompt_enhancement.strip()

